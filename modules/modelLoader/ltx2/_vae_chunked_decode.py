"""OT-resident chunked streaming decode for the diffusers LTX-2.3 video VAE.

Goal: replicate ComfyUI's internal streaming decode (a single low-VRAM pass over
temporal chunks with a per-conv rolling cache, written into a preallocated output
buffer) so ``vae.decode`` no longer routes through diffusers' overlap-blend tiling
(``_temporal_tiled_decode`` / ``tiled_decode``) which materialises the whole temporal
volume (``list`` + ``torch.cat``) and is 20-200x slower / spills to shared VRAM.

Lives ENTIRELY in OT and patches the *loaded* VAE module instances' ``forward`` at
runtime — no edit to the diffusers venv package, no vendoring of ComfyUI (GPL). This
mirrors OT's existing runtime-patch pattern (``_diffusers_patch.py``,
``_ffn_chunk_patch.py``, ``_sequential_cfg_patch.py``) and the per-instance
``module.forward = ...`` swap used by the pinned-LoRA path.

STATUS / SCOPE (see research/ltx23_vae_chunked_decode_plan.md):
  * The LTX-2.3 VAE config is ``decoder_causal=False`` — a NON-causal decoder. Its
    causal-conv layers pad symmetrically, so faithful streaming needs (a) a
    sliding-window conv cache (output lags by the right-context until flushed on the
    last chunk) and (b) residual alignment so each resnet/upsampler skip-add lines up
    with the lagged conv-branch output.
  * (a) the sliding-window conv cache + the streaming DRIVER + infra below are
    self-contained and correct for both causal and non-causal.
  * (b) residual alignment (resnet / upsampler skip paths) is FIFO-aligned to the
    lagged conv-branch output (non-causal case). Validated against a bit-parity gate
    (``OneTrainerSampler/scripts/vae_parity_check.py``): no-chunk == full decode (max|Δ|=0);
    with chunking fp32 ~2e-3 (uniform cuDNN-shape noise), bf16 ~1 ULP.

This is now the DEFAULT LTX-2.3 VAE decode path. Any shape/parity mismatch raises so the
caller (``Ltx2Sampler._decode_video_with_oom_fallback``) falls back to the legacy tiled
decode. Escape hatch: ``LTX2_VAE_CHUNKED_DECODE=0`` forces the legacy path. Debug:
``LTX2_VAE_STREAM_DEBUG=1`` / ``LTX2_VAE_CONV_DEBUG=1``.
"""
from __future__ import annotations

import os
import time

import torch


# ---------------------------------------------------------------------------
# Capability + identification helpers
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Chunked streaming decode is the DEFAULT LTX-2.3 VAE path. Escape hatch:
    set ``LTX2_VAE_CHUNKED_DECODE=0`` to force the legacy tiled decode."""
    return os.environ.get("LTX2_VAE_CHUNKED_DECODE", "1").strip() not in ("0", "false", "False")


def _chunk_frames_env(default: int) -> int:
    try:
        v = int(os.environ.get("LTX2_VAE_CHUNK_FRAMES", "").strip())
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _clsname(m) -> str:
    return type(m).__name__


def _is_causal_conv(m) -> bool:
    return _clsname(m) == "LTX2VideoCausalConv3d" and hasattr(m, "conv") and hasattr(m, "kernel_size")


def _is_upsampler(m) -> bool:
    return _clsname(m) == "LTX2VideoUpsampler3d"


def _is_resnet(m) -> bool:
    return _clsname(m) == "LTX2VideoResnetBlock3d"


class StreamingUnsupported(RuntimeError):
    """Raised when the loaded VAE doesn't match the assumptions this path makes."""


def assert_supported(vae) -> None:
    """Fail fast (→ caller falls back) if the VAE is outside the supported subset.

    Supported subset (matches the LTX-2.3 ``dg845`` config we target):
      * a ``decoder`` submodule with the expected LTX2 module classes
      * timestep_conditioning OFF (no per-resnet scale_shift_table)
      * inject_noise OFF (no per_channel_scale*) — its per-chunk RNG would break parity
      * patch_size_t == 1 (no temporal patch folding)
    """
    dec = getattr(vae, "decoder", None)
    if dec is None:
        raise StreamingUnsupported("vae has no .decoder")
    if int(getattr(dec, "patch_size_t", 1)) != 1:
        raise StreamingUnsupported("patch_size_t != 1 not supported")
    n_conv = 0
    for m in dec.modules():
        if _is_causal_conv(m):
            n_conv += 1
        if _is_resnet(m):
            if getattr(m, "scale_shift_table", None) is not None:
                raise StreamingUnsupported("resnet timestep_conditioning not supported")
            if getattr(m, "per_channel_scale1", None) is not None or getattr(m, "per_channel_scale2", None) is not None:
                raise StreamingUnsupported("resnet inject_noise not supported (per-chunk RNG breaks parity)")
    if n_conv == 0:
        raise StreamingUnsupported("no LTX2VideoCausalConv3d found in decoder")


# ---------------------------------------------------------------------------
# Streaming context shared by all patched modules for one decode
# ---------------------------------------------------------------------------

class _StreamCtx:
    # ``ended`` mirrors ComfyUI's per-conv ``is_end`` flag, set right before each stage
    # call in the depth-first driver (only one path is active at a time). The convs read
    # it to decide whether to flush (right-pad) on the final chunk of a path.
    __slots__ = ("ended",)

    def __init__(self) -> None:
        self.ended = False


# ---------------------------------------------------------------------------
# (a) sliding-window cache-aware causal-conv forward  [self-contained, both modes]
# ---------------------------------------------------------------------------

def _make_conv_forward(conv_mod, ctx: _StreamCtx):
    """Replacement forward for ``LTX2VideoCausalConv3d`` with a rolling temporal cache.

    Reproduces diffusers' per-call temporal padding (causal: left k-1; non-causal:
    symmetric (k-1)//2 each side) but carries the boundary across chunks via a cache
    so the streamed result equals a single full forward — no overlap-blend.

    Spatial padding is left to ``conv_mod.conv`` (its nn.Conv3d padding=(0,h,w)); we
    only manage the temporal dim (dim=2), exactly like the original which temporally
    pads then calls ``self.conv`` with temporal padding 0.
    """
    conv = conv_mod.conv
    k = conv_mod.kernel_size[0]

    def forward(hidden_states, causal: bool = True):
        if k == 1:  # 1x1x1: no temporal mixing, nothing to cache
            return conv(hidden_states)
        cache = getattr(conv_mod, "_ot_tcache", None)
        if cache is None:
            pad_len = (k - 1) if causal else (k - 1) // 2
            if hidden_states.shape[2] == 0:
                return hidden_states
            left = hidden_states[:, :, :1, :, :].repeat((1, 1, pad_len, 1, 1))
            pieces = [left, hidden_states]
        else:
            pieces = [cache, hidden_states]
        if ctx.ended and not causal:
            # flush: supply the trailing right-context the symmetric padding needs
            pieces.append(hidden_states[:, :, -1:, :, :].repeat((1, 1, (k - 1) // 2, 1, 1)))
        xin = torch.cat(pieces, dim=2)
        # cache the last (k - stride) frames of the assembled input for the next chunk
        stride_t = conv.stride[0] if isinstance(conv.stride, (tuple, list)) else conv.stride
        cache_len = k - stride_t
        if not ctx.ended:
            # .clone() is essential: a plain slice is a VIEW that pins the ENTIRE
            # assembled input tensor in VRAM. With ~40 convs each pinning their
            # (high-res) input, that alone balloons peak VRAM by tens of GB. We only
            # need the last `cache_len` frames — copy them out.
            conv_mod._ot_tcache = xin[:, :, -cache_len:, :, :].detach().clone()
        else:
            conv_mod._ot_tcache = None
        if xin.shape[2] < k:
            return xin[:, :, :0, :, :]
        if os.environ.get("LTX2_VAE_CONV_DEBUG"):
            print(f"[conv] in={tuple(xin.shape)} w={tuple(conv.weight.shape)} "
                  f"stride={conv.stride}")
        return conv(xin)

    return forward


# ---------------------------------------------------------------------------
# (b) residual alignment via per-module FIFO
# ---------------------------------------------------------------------------
# A non-causal symmetric conv is length-preserving in the full decode, so the
# streamed conv emits the SAME output frames, only delayed. Both the conv branch
# and the skip/residual branch therefore start at global frame 0 and advance in
# order — aligning them is a plain FIFO: buffer the residual frames and release
# exactly as many as the conv branch emitted this chunk. The buffer holds the
# per-layer lag between chunks and flushes on the last chunk.

def _fifo_push_take(module, attr: str, frames: torch.Tensor, n_take: int) -> torch.Tensor:
    buf = getattr(module, attr, None)
    if buf is not None and buf.shape[2] > 0:
        frames = torch.cat([buf, frames], dim=2)
    if frames.shape[2] < n_take:
        raise StreamingUnsupported(
            f"residual FIFO underflow on {attr}: have {frames.shape[2]} need {n_take}")
    take = frames[:, :, :n_take, :, :]
    rest = frames[:, :, n_take:, :, :]
    # .clone() the retained buffer — a view would pin the whole concatenated tensor.
    setattr(module, attr, rest.detach().clone() if rest.shape[2] > 0 else None)
    return take


def _make_resnet_forward(resnet, ctx: _StreamCtx):
    """Streaming forward for LTX2VideoResnetBlock3d (timestep/inject_noise OFF subset).

    Identical math to diffusers' forward except the residual add is FIFO-aligned to
    the (lagged) conv-branch output. conv1/conv2 are the patched cache-aware convs.
    """
    def forward(inputs, temb=None, generator=None, causal: bool = True):
        h = resnet.norm1(inputs)
        h = resnet.nonlinearity(h)
        h = resnet.conv1(h, causal=causal)
        h = resnet.norm2(h)
        h = resnet.nonlinearity(h)
        h = resnet.dropout(h)
        h = resnet.conv2(h, causal=causal)

        res = inputs
        if resnet.norm3 is not None:
            res = resnet.norm3(res.movedim(1, -1)).movedim(-1, 1)
        if resnet.conv_shortcut is not None:
            res = resnet.conv_shortcut(res)  # 1x1x1, temporally length-preserving
        res = _fifo_push_take(resnet, "_ot_res_fifo", res, h.shape[2])
        return h + res

    return forward


def _make_upsampler_forward(ups, ctx: _StreamCtx):
    """Streaming forward for LTX2VideoUpsampler3d.

    Mirrors diffusers' forward but (1) reshapes the main branch by the conv's ACTUAL
    output frame count (which lags under streaming), (2) applies the leading
    ``stride0-1`` temporal trim only on the first chunk (it drops the global warmup
    frame, not a per-chunk one), and (3) FIFO-aligns the residual branch to the main
    branch's frame count.
    """
    s0, s1, s2 = ups.stride

    def forward(hidden_states, causal: bool = True):
        # Streaming warmup can drive a chunk's temporal length to 0 at this depth
        # (the cache will emit those frames on later chunks). All reshapes are guarded
        # for 0-frame so empty chunks flow through harmlessly.
        b, c, n, h, w = hidden_states.shape
        res = None
        if ups.residual and n > 0:
            res = hidden_states.reshape(b, -1, s0, s1, s2, n, h, w)
            res = res.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
            repeats = (s0 * s1 * s2) // ups.upscale_factor
            res = res.repeat(1, repeats, 1, 1, 1)

        conv_out = ups.conv(hidden_states, causal=causal)
        m = conv_out.shape[2]
        if m > 0:
            main = conv_out.reshape(b, -1, s0, s1, s2, m, conv_out.shape[3], conv_out.shape[4])
            main = main.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        else:
            c_out = conv_out.shape[1]
            main = conv_out.new_zeros((b, c_out // (s0 * s1 * s2), 0, h * s1, w * s2))

        if ups.residual:
            if res is None:
                res = main.new_zeros((b, main.shape[1], 0, main.shape[3], main.shape[4]))
            res = _fifo_push_take(ups, "_ot_ures_fifo", res, main.shape[2])
            main = main + res

        # Leading temporal trim (drops the global warmup frames). Diffusers trims each
        # branch by s0-1 then adds == trimming (main+res) by s0-1. Done as a CUMULATIVE
        # counter over the output stream so it lands on the first frames actually
        # emitted, robust to streaming warmup delaying them past the first input chunk.
        trim = getattr(ups, "_ot_trim_remaining", s0 - 1)
        if trim > 0 and main.shape[2] > 0:
            cut = min(trim, main.shape[2])
            main = main[:, :, cut:]
            ups._ot_trim_remaining = trim - cut
        return main

    return forward


# ---------------------------------------------------------------------------
# Install / reset / remove
# ---------------------------------------------------------------------------

class StreamingHandle:
    """Tracks patched modules + the shared ctx; restores originals on remove()."""

    def __init__(self, vae, ctx: _StreamCtx):
        self.vae = vae
        self.ctx = ctx
        self._originals: list[tuple[object, object]] = []  # (module, original_forward)
        self._conv_mods: list[object] = []
        self._fifo_mods: list[tuple[object, str]] = []  # (module, fifo_attr)
        self._trim_mods: list[tuple[object, int]] = []   # (upsampler, initial_trim)

    def _patch(self, module, new_forward):
        self._originals.append((module, module.forward))
        module.forward = new_forward

    def reset(self) -> None:
        """Clear all per-module caches/FIFOs/trim counters for a fresh decode."""
        for m in self._conv_mods:
            if hasattr(m, "_ot_tcache"):
                m._ot_tcache = None
        for m, attr in self._fifo_mods:
            setattr(m, attr, None)
        for m, init in self._trim_mods:
            m._ot_trim_remaining = init
        self.ctx.ended = False

    def remove(self) -> None:
        for module, original in reversed(self._originals):
            try:
                module.forward = original
            except (AttributeError, TypeError):
                pass
        for m in self._conv_mods:
            if hasattr(m, "_ot_tcache"):
                try:
                    del m._ot_tcache
                except AttributeError:
                    pass
        for m, attr in self._fifo_mods:
            if hasattr(m, attr):
                try:
                    delattr(m, attr)
                except AttributeError:
                    pass
        for m, _init in self._trim_mods:
            if hasattr(m, "_ot_trim_remaining"):
                try:
                    delattr(m, "_ot_trim_remaining")
                except AttributeError:
                    pass
        self._originals.clear()
        self._conv_mods.clear()
        self._fifo_mods.clear()
        self._trim_mods.clear()


def install_streaming(vae, ctx: _StreamCtx) -> StreamingHandle:
    """Patch the decoder for streaming: cache-aware convs + FIFO-aligned residuals.

    Patches every ``LTX2VideoCausalConv3d`` (sliding-window temporal cache),
    ``LTX2VideoResnetBlock3d`` and ``LTX2VideoUpsampler3d`` (residual FIFO alignment).
    The block wrappers (mid/up blocks) need no patching — they just call the leaf
    modules, which are patched. Validated against the bit-parity gate.
    """
    handle = StreamingHandle(vae, ctx)
    for m in vae.decoder.modules():
        if _is_causal_conv(m):
            m._ot_tcache = None
            handle._conv_mods.append(m)
            handle._patch(m, _make_conv_forward(m, ctx))
        elif _is_resnet(m):
            m._ot_res_fifo = None
            handle._fifo_mods.append((m, "_ot_res_fifo"))
            handle._patch(m, _make_resnet_forward(m, ctx))
        elif _is_upsampler(m):
            m._ot_ures_fifo = None
            init_trim = int(m.stride[0]) - 1  # frames the upsampler drops once (drop_first)
            m._ot_trim_remaining = init_trim
            handle._fifo_mods.append((m, "_ot_ures_fifo"))
            handle._trim_mods.append((m, init_trim))
            handle._patch(m, _make_upsampler_forward(m, ctx))
    return handle


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _decode_output_shape(vae, z: torch.Tensor) -> tuple[int, int, int, int, int]:
    b, _c, t, h, w = z.shape
    tc = int(vae.temporal_compression_ratio)
    sc = int(vae.spatial_compression_ratio)
    out_ch = int(getattr(vae.config, "out_channels", 3)) if hasattr(vae, "config") else 3
    t_px = (t - 1) * tc + 1
    return b, out_ch, t_px, h * sc, w * sc


def _max_chunk_bytes(device) -> int:
    """Port of ComfyUI ``get_max_chunk_size``: 32 MiB (<=6 GB VRAM) .. 128 MiB (>=24 GB),
    linearly interpolated. Bounds the per-stage activation working set."""
    lo, hi = 32 * 1024 * 1024, 128 * 1024 * 1024
    try:
        _free, total = torch.cuda.mem_get_info(device)
        gb = total / (1024 ** 3)
    except Exception:
        return lo
    if gb <= 6:
        return lo
    if gb >= 24:
        return hi
    return int(lo + (gb - 6) / 18.0 * (hi - lo))


def _decode_tail(decoder, sample, causal, ctx, ended, out, offset) -> None:
    """norm_out + act + conv_out + unpatchify for one chunk, written into ``out``.

    Mirrors the tail of ComfyUI ``run_up`` (idx past the last up-block). Skips the
    timestep ada-shift (``assert_supported`` excludes timestep VAEs). 0-frame-safe.
    """
    ctx.ended = ended
    sample = decoder.norm_out(sample)
    sample = decoder.conv_act(sample)
    sample = decoder.conv_out(sample, causal=causal)
    if sample is None or sample.shape[2] == 0:
        return
    p, p_t = decoder.patch_size, decoder.patch_size_t
    b, c, n, hh, ww = sample.shape
    sample = sample.reshape(b, -1, p_t, p, p, n, hh, ww)
    sample = sample.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    t = sample.shape[2]
    if offset[0] + t > out.shape[2]:
        raise StreamingUnsupported(
            f"output overflow: off={offset[0]} t={t} cap={out.shape[2]}")
    out[:, :, offset[0]:offset[0] + t].copy_(sample.to(out.device, out.dtype))
    offset[0] += t


def _run_up(decoder, stages, idx, sample, ended, ctx, out, offset, max_chunk, causal) -> None:
    """Depth-first streaming over decoder stages — a port of ComfyUI ``Decoder.run_up``.

    Run ``stages[idx]`` on the FULL current tensor (cheap early stages stay whole),
    then split its OUTPUT temporally by the byte budget and recurse per sub-chunk; the
    patched leaf modules carry continuity across the sub-chunk calls via their caches.
    A 0-frame stage output ends this branch (later sub-chunks emit those frames). The
    final chunk of each path carries ``ended=True`` so the convs flush (right-pad).
    """
    if idx >= len(stages):
        _decode_tail(decoder, sample, causal, ctx, ended, out, offset)
        return
    ctx.ended = ended
    _in_f = sample.shape[2]
    sample = stages[idx](sample)
    if sample is None or sample.shape[2] == 0:
        return
    total_bytes = sample.numel() * sample.element_size()
    num_chunks = max(1, (total_bytes + max_chunk - 1) // max_chunk)
    if os.environ.get("LTX2_VAE_STREAM_DEBUG"):
        print(f"[stream] stage={idx} in_f={_in_f} out_shape={tuple(sample.shape)} "
              f"out={total_bytes/1024**3:.2f}GB num_chunks={num_chunks} ended={ended}")
    if num_chunks == 1:
        _run_up(decoder, stages, idx + 1, sample, ended, ctx, out, offset, max_chunk, causal)
    else:
        subs = torch.chunk(sample, num_chunks, dim=2)
        for j, sub in enumerate(subs):
            _run_up(decoder, stages, idx + 1, sub, ended and (j == len(subs) - 1),
                    ctx, out, offset, max_chunk, causal)


@torch.no_grad()
def chunked_decode(vae, z: torch.Tensor, *, causal: bool | None = None,
                   output_device=None, output_dtype=None,
                   max_chunk_bytes: int | None = None) -> torch.Tensor:
    """Stream-decode ``z`` into one preallocated buffer (ComfyUI ``run_up`` structure).

    ``conv_in`` and each stage run on the full current tensor; the OUTPUT between
    stages is split by a byte budget and recursed depth-first so only one sub-chunk's
    worth of high-resolution activation is resident at a time. Raises on any
    frame-count mismatch so the caller can fall back to the stock decode.
    """
    assert_supported(vae)
    decoder = vae.decoder
    causal = causal if causal is not None else bool(getattr(decoder, "is_causal", False))

    # Default the output buffer to CPU (ComfyUI's intermediate_device): the full decoded
    # video is large and holding it in VRAM eats into the budget cuDNN needs for the conv
    # workspaces. Chunks are copied to the CPU buffer as they're produced.
    out_device = output_device if output_device is not None else torch.device("cpu")
    out_dtype = output_dtype if output_dtype is not None else z.dtype
    b, out_ch, t_px, h_px, w_px = _decode_output_shape(vae, z)
    out = torch.empty((b, out_ch, t_px, h_px, w_px), device=out_device, dtype=out_dtype)

    ctx = _StreamCtx()
    handle = install_streaming(vae, ctx)
    handle.reset()
    offset = [0]
    max_chunk = max_chunk_bytes or _max_chunk_bytes(z.device)

    # Flatten the decoder into LEAF stages (individual resnets / upsamplers) in
    # execution order, so the byte-budget chunking in _run_up happens BETWEEN each leaf
    # — mirroring ComfyUI's flat run_up block list. This is what bounds memory: after an
    # upsampler doubles the frame count, its output is chunked before the resnets, so a
    # resnet conv never processes the full (e.g. 51-frame) high-res volume at once.
    def _resnet_stage(r):
        return lambda s: r(s, None, causal=causal)

    def _upsampler_stage(u):
        return lambda s: u(s, causal=causal)

    stages = []
    for r in decoder.mid_block.resnets:
        stages.append(_resnet_stage(r))
    for ub in decoder.up_blocks:
        if getattr(ub, "conv_in", None) is not None:
            stages.append(_resnet_stage(ub.conv_in))
        if getattr(ub, "upsamplers", None) is not None:
            for u in ub.upsamplers:
                stages.append(_upsampler_stage(u))
        for r in ub.resnets:
            stages.append(_resnet_stage(r))
    # cuDNN conv3d has a known bug where its heuristic picks an FFT-based algorithm
    # needing a giant (10s of GB) workspace for video-VAE shapes. With benchmark=False
    # the heuristic returns that algo and PyTorch tries it with no fallback → OOM. With
    # benchmark=True (what ComfyUI uses), cuDNN trials algorithms, catches per-algo OOM,
    # and falls back to a fitting low-workspace algo. So we force benchmark ON here.
    _bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    t0 = time.perf_counter()
    try:
        # conv_in on the full (cheap, latent-res) input as a single flushed pass.
        ctx.ended = True
        h = decoder.conv_in(z, causal=causal)
        _run_up(decoder, stages, 0, h, True, ctx, out, offset, max_chunk, causal)
        if offset[0] != t_px:
            raise StreamingUnsupported(
                f"streamed {offset[0]} frames, expected {t_px}")
    finally:
        torch.backends.cudnn.benchmark = _bench
        handle.remove()
    # Reaching here means streaming succeeded (errors raise → caller falls back to tiled).
    print(f"[Ltx2 VAE] streaming decode: {z.shape[2]}→{t_px}f @ {h_px}x{w_px}, "
          f"{len(stages)} leaf-stages, chunk≤{max_chunk // (1024 * 1024)}MB, "
          f"out→{out_device}, {time.perf_counter() - t0:.1f}s")
    return out
