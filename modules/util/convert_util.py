from collections.abc import Callable
from dataclasses import dataclass

import torch

import parse


@dataclass
class ConversionPattern:
    from_patterns: list[str]
    to_patterns: list[str]
    convert_fn: Callable | None
    reverse_convert_fn: Callable | None
    children : list["ConversionPattern"]


def _convert_item(in_key: str, input: dict, conversions: list[ConversionPattern], in_prefix: str="", out_prefix: str="", in_separator='.', out_separator='.'):
    for conversion in conversions:
        if conversion.children:
            if len(conversion.from_patterns) > 1:
                raise RuntimeError("Only leafs can have multiple from-patterns")
            if len(conversion.to_patterns) > 1:
                raise RuntimeError("Only leafs can have multiple to-patterns")

            match = parse.parse(in_prefix + conversion.from_patterns[0] + in_separator + "{post__}", in_key)
            if match is None:
                continue
            child_in_prefix = in_prefix + conversion.from_patterns[0].format(*match.fixed, **match.named) + in_separator
            child_out_prefix = out_prefix + conversion.to_patterns[0].format(*match.fixed, **match.named) + out_separator
            return _convert_item(in_key, input, conversion.children, in_prefix=child_in_prefix, out_prefix=child_out_prefix, in_separator=in_separator, out_separator=out_separator)
        else:
            for pattern in conversion.from_patterns:
                match = parse.parse(in_prefix + pattern, in_key)
                if match is not None:
                    break

            if match is None:
                for pattern in conversion.from_patterns:
                    match =  parse.parse(in_prefix + pattern + in_separator + "{post__}", in_key)
                    if match is not None:
                        break
                if match is None:
                    continue
                in_postfix = in_separator + match.named['post__']
                out_postfix = out_separator + match.named['post__']
            else:
                in_postfix = ""
                out_postfix = ""

            in_keys = []
            in_values = []
            try:
                for pattern in conversion.from_patterns:
                    new_in_key = in_prefix + pattern.format(*match.fixed, **match.named) + in_postfix
                    in_keys.append(new_in_key)
                    in_values.append(input[new_in_key])
            except KeyError:
                #not a match, because not all from_patterns were found:
                continue

            out_keys = [out_prefix + pattern.format(*match.fixed, **match.named) + out_postfix for pattern in conversion.to_patterns]
            if conversion.convert_fn is not None:
                out_values = conversion.convert_fn(*in_values)
                if not isinstance(out_values, tuple):
                    out_values = (out_values, )

                if len(out_values) != len(out_keys):
                    raise RuntimeError("convert_fn returned invalid number of outputs, for key " + in_key)
                return in_keys, dict(zip(out_keys, out_values, strict=True))
            else:
                if len(out_keys) > 1:
                    raise RuntimeError("A convert_fn must be provided if there are multiple to-patterns")
                if len(in_keys) > 1:
                    raise RuntimeError("A convert_fn must be provided if there are multiple in-patterns")
                return in_keys, {
                    out_keys[0]: in_values[0],
                }

    return [in_key], None

def _is_conversion_pattern_list(conversions: list):
    return all(isinstance(entry, ConversionPattern) for entry in conversions)

def _is_tuple_list(input: list):
    return isinstance(input, list) and all(isinstance(entry, tuple) for entry in input)

def _create_conversions_list(conversion_input: list):
    if _is_tuple_list(conversion_input):
        conversion_input = [conversion_input]
    output = []
    for entry in conversion_input:
        if _is_tuple_list(entry):
            entry = _create_conversion_from_tuple_list(entry)
        if _is_conversion_pattern_list(entry):
            output.append(entry)
        else:
            raise RuntimeError("conversion input is invalid")
    return output


def convert(input_orig: dict, conversion_input: list[ConversionPattern] | list, strict: bool=True, in_separator='.', out_separator='.'):
    conversions_list = _create_conversions_list(conversion_input)

    input = input_orig.copy()
    for conversions in conversions_list:
        output = {}
        while len(input) > 0:
            in_key = next(iter(input))
            input_keys, output_items = _convert_item(in_key, input, conversions, in_separator=in_separator, out_separator=out_separator)
            if output_items is None:
                if strict:
                    raise RuntimeError("No conversion found for key " + in_key)
                if in_key in output and not output[in_key].equal(input[in_key]):
                    raise RuntimeError(f"key {in_key} was generated twice during conversion and is not equal")
                output[in_key] = input[in_key]
            else:
                for k, v in output_items.items():
                    if k in output and not torch.equal(v, output[k]):
                        raise RuntimeError(f"key {k} was generated twice during conversion and is not equal")

                output |= output_items
            for k in input_keys:
                input.pop(k)

        assert len(input) == 0
        input = output

    return output


def reverse_conversion_pattern(input: ConversionPattern):
    if input.convert_fn is not None and input.reverse_convert_fn is None:
        raise RuntimeError("Conversion cannot be reversed: no reverse_convert_fn defined")

    return ConversionPattern(
        from_patterns=input.to_patterns,
        to_patterns=input.from_patterns,
        convert_fn=input.reverse_convert_fn,
        reverse_convert_fn=input.convert_fn,
        children=reverse_conversion(input.children) if input.children else None,
    )

def reverse_conversion(input: list[ConversionPattern]):
    return [reverse_conversion_pattern(entry) for entry in input]

def _create_pattern_list(input: str | list[str]):
    pattern = input
    if isinstance(pattern, str):
        pattern = [pattern]
    if not isinstance(pattern, list) or any(not isinstance(f, str) for f in pattern):
        raise ValueError("conversion pattern must either be a string, or a list of strings")
    return pattern


def _create_conversion_pattern_from_tuple(input: tuple | ConversionPattern):
    if isinstance(input, ConversionPattern):
        return input
    if not isinstance(input, tuple) or len(input) < 2:
        raise ValueError("conversion entry must be a tuple of at least 2 items")

    from_patterns = _create_pattern_list(input[0])
    if isinstance(input[1], list) and all(isinstance(entry, tuple) for entry in input[1]):
        children_in = input[1]
        to_patterns = from_patterns
    else:
        to_patterns = _create_pattern_list(input[1])
        children_in = input[2] if len(input) > 2 and isinstance(input[2], list) else None

    convert_fn = None
    reverse_convert_fn = None
    children = None
    if children_in is not None:
        children = _create_conversion_from_tuple_list(children_in)
    elif len(input) > 2:
        convert_fn = input[2]
        reverse_convert_fn = input[3] if len(input) > 3 else None

    if (len(from_patterns) > 1 or len(to_patterns) > 1) and convert_fn is None:
        raise ValueError("conversion entries with more than one to- or from-pattern require a convert function")

    return ConversionPattern(from_patterns, to_patterns, convert_fn, reverse_convert_fn, children)

def _create_conversion_from_tuple_list(input: list):
    return [_create_conversion_pattern_from_tuple(entry) for entry in input]

def fuse_qkv(q, k, v):
    return torch.cat([q, k, v], dim=0)

def fuse_qkv_mlp(q, k, v, mlp):
    return torch.cat([q, k, v, mlp], dim=0)


def remove_prefix(prefix: str | None = None, separator: str='.'):
    if prefix is None:
        prefix = "{prefix__}"
    return [(prefix + separator + "{key}", "{key}")]

def add_prefix(prefix: str, separator: str='.'):
    return [("{}", prefix + separator + "{}")]

def lora_fuse_qkv(q_up, q_down, q_alpha, k_up, k_down, k_alpha, v_up, v_down, v_alpha):
    dim, rank = q_up.shape
    qkv_up = torch.zeros(
        3 * dim,
        3 * rank,
        device=q_up.device,
        dtype=q_up.dtype,
    )
    qkv_up[dim*0:dim*1, rank*0:rank*1] = q_up
    qkv_up[dim*1:dim*2, rank*1:rank*2] = k_up
    qkv_up[dim*2:dim*3, rank*2:rank*3] = v_up
    qkv_down = torch.cat([q_down, k_down, v_down], dim=0)

    qkv_alpha = q_alpha * 3
    if q_alpha != k_alpha or q_alpha != v_alpha:
        raise NotImplementedError("fused layers must have the same alpha")

    return qkv_up, qkv_down, qkv_alpha

def lora_fuse_qkv_mlp(q_up, q_down, q_alpha, k_up, k_down, k_alpha, v_up, v_down, v_alpha, mlp_up, mlp_down, mlp_alpha):
    dim, rank = q_up.shape
    mlp_dim = mlp_up.shape[0]
    qkv_up = torch.zeros(
        3 * dim + mlp_dim,
        4 * rank,
        device=q_up.device,
        dtype=q_up.dtype,
    )
    qkv_up[dim*0:dim*1, rank*0:rank*1] = q_up
    qkv_up[dim*1:dim*2, rank*1:rank*2] = k_up
    qkv_up[dim*2:dim*3, rank*2:rank*3] = v_up
    qkv_up[dim*3:,      rank*3:rank*4] = mlp_up
    qkv_down = torch.cat([q_down, k_down, v_down, mlp_down], dim=0)

    qkv_alpha = q_alpha * 4
    if q_alpha != k_alpha or q_alpha != v_alpha or q_alpha != mlp_alpha:
        raise NotImplementedError("fused layers must have the same alpha")

    return qkv_up, qkv_down, qkv_alpha

def lora_fuse_qkv_to_qkv_mlp(q_up, q_down, q_alpha, k_up, k_down, k_alpha, v_up, v_down, v_alpha):
    #TODO where to get output shape from, if there is no MLP dim?
    raise NotImplementedError

def lora_fuse_mlp_to_qkv_mlp(mlp_up, mlp_down, mlp_alpha):
    #TODO where to get output shape from, if there is no qkv dim?
    raise NotImplementedError


def lora_unfuse_qkv(qkv_up, qkv_down, qkv_alpha):
    """Inverse of lora_fuse_qkv. Splits OT's block-diagonal QKV adapter back into Q, K, V.

    Recovers Q, K, V by extracting the three diagonal blocks from qkv_up and splitting
    qkv_down into three equal parts. Exact round-trip with lora_fuse_qkv.
    Input must be block-diagonal as produced by lora_fuse_qkv (not a native fused adapter).
    For native fused adapters use lora_split_qkv instead.
    """
    if qkv_down.shape[0] % 3 != 0:
        raise RuntimeError(f"lora_unfuse_qkv: qkv_down row count {qkv_down.shape[0]} is not divisible by 3")
    if qkv_up.shape[0] % 3 != 0:
        raise RuntimeError(f"lora_unfuse_qkv: qkv_up row count {qkv_up.shape[0]} is not divisible by 3")
    rank = qkv_down.shape[0] // 3
    dim  = qkv_up.shape[0]  // 3
    q_up   = qkv_up[0:dim,       0:rank      ]
    k_up   = qkv_up[dim:2*dim,   rank:2*rank ]
    v_up   = qkv_up[2*dim:3*dim, 2*rank:3*rank]
    q_down = qkv_down[0:rank      ]
    k_down = qkv_down[rank:2*rank ]
    v_down = qkv_down[2*rank:3*rank]
    alpha  = qkv_alpha / 3
    return q_up, q_down, alpha, k_up, k_down, alpha, v_up, v_down, alpha


def lora_unfuse_qkv_mlp(qkv_up, qkv_down, qkv_alpha):
    """Inverse of lora_fuse_qkv_mlp. Splits OT's block-diagonal QKVM adapter back into Q, K, V, MLP.

    Recovers per-component output dims from the block-diagonal structure: column block i
    (width rank) is non-zero only in the rows belonging to its component. Component dims
    may be unequal (e.g. Q/K/V dim != MLP dim).
    Input must be block-diagonal as produced by lora_fuse_qkv_mlp.
    Exact round-trip with lora_fuse_qkv_mlp.
    """
    if qkv_down.shape[0] % 4 != 0:
        raise RuntimeError(f"lora_unfuse_qkv_mlp: qkv_down row count {qkv_down.shape[0]} is not divisible by 4")
    rank = qkv_down.shape[0] // 4

    def _row_range(col_start: int) -> tuple[int, int]:
        block = qkv_up[:, col_start:col_start + rank]
        nonzero = block.abs().sum(dim=1).nonzero(as_tuple=False).flatten()
        if nonzero.numel() == 0:
            return 0, 0
        return int(nonzero[0].item()), int(nonzero[-1].item()) + 1

    q_start, q_end = _row_range(0)
    k_start, k_end = _row_range(rank)
    v_start, v_end = _row_range(2 * rank)
    m_start, m_end = _row_range(3 * rank)

    q_up   = qkv_up[q_start:q_end, 0:rank      ]
    k_up   = qkv_up[k_start:k_end, rank:2*rank ]
    v_up   = qkv_up[v_start:v_end, 2*rank:3*rank]
    mlp_up = qkv_up[m_start:m_end, 3*rank:4*rank]

    q_down   = qkv_down[0:rank      ]
    k_down   = qkv_down[rank:2*rank ]
    v_down   = qkv_down[2*rank:3*rank]
    mlp_down = qkv_down[3*rank:4*rank]

    alpha = qkv_alpha / 4
    return q_up, q_down, alpha, k_up, k_down, alpha, v_up, v_down, alpha, mlp_up, mlp_down, alpha


def lora_split_qkv(qkv_up, qkv_down, qkv_alpha):
    """Split a natively-fused QKV LoRA adapter into separate Q, K, V adapters.

    For adapters trained on the full fused QKV projection (lora_up shape (3*dim, rank)),
    where all three projections share a single lora_down. Slices lora_up along dim 0.
    The result is mathematically exact: (alpha/rank) * up_q @ down == delta_W_Q.

    Use when loading natively-trained (non-OT) fused QKV LoRAs.
    Distinguish from lora_unfuse_qkv (for OT block-diagonal exports) by checking
    lora_up.shape[1]: rank → native, 3*rank → block-diagonal.

    Inverse: lora_join_qkv (requires identical lora_down and alpha across Q/K/V).
    """
    if qkv_up.shape[0] % 3 != 0:
        raise RuntimeError(f"lora_split_qkv: qkv_up row count {qkv_up.shape[0]} is not divisible by 3")
    dim = qkv_up.shape[0] // 3
    q_up = qkv_up[0:dim,       :]
    k_up = qkv_up[dim:2*dim,   :]
    v_up = qkv_up[2*dim:3*dim, :]
    return q_up, qkv_down, qkv_alpha, k_up, qkv_down, qkv_alpha, v_up, qkv_down, qkv_alpha


def lora_join_qkv(q_up, q_down, q_alpha, k_up, k_down, k_alpha, v_up, v_down, v_alpha):
    """Join separate Q/K/V adapters back into a natively-fused QKV adapter.

    Requires identical lora_down and alpha across Q, K, V — guaranteed after lora_split_qkv.
    """
    if not (torch.equal(q_down, k_down) and torch.equal(q_down, v_down)):
        raise ValueError("lora_join_qkv requires identical lora_down for Q, K, V")
    if q_alpha != k_alpha or q_alpha != v_alpha:
        raise ValueError("lora_join_qkv requires identical alpha for Q, K, V")
    return torch.cat([q_up, k_up, v_up], dim=0), q_down, q_alpha


def lora_split_qkv_mlp(qkvm_up, qkvm_down, qkvm_alpha):
    """Split a natively-fused QKVM LoRA adapter into separate Q, K, V, MLP adapters.

    For adapters trained on the full fused Q/K/V/MLP projection (lora_up shape
    (q_dim+k_dim+v_dim+mlp_dim, rank)) where all components share a single lora_down.

    Requires equal output dimensions for all four components. If the model has unequal
    component dims (e.g. HunyuanVideo where proj_mlp has a different hidden size than Q/K/V),
    use a model-specific wrapper that provides the correct split indices instead of this function.

    Inverse: lora_join_qkv_mlp (requires identical lora_down and alpha across all components).
    """
    if qkvm_up.shape[0] % 4 != 0:
        raise RuntimeError(
            f"lora_split_qkv_mlp: qkvm_up row count {qkvm_up.shape[0]} is not divisible by 4. "
            "For models with unequal component dims, use a model-specific split wrapper."
        )
    dim = qkvm_up.shape[0] // 4
    q_up   = qkvm_up[0:dim,       :]
    k_up   = qkvm_up[dim:2*dim,   :]
    v_up   = qkvm_up[2*dim:3*dim, :]
    mlp_up = qkvm_up[3*dim:,      :]
    return (q_up,   qkvm_down, qkvm_alpha,
            k_up,   qkvm_down, qkvm_alpha,
            v_up,   qkvm_down, qkvm_alpha,
            mlp_up, qkvm_down, qkvm_alpha)


def lora_join_qkv_mlp(q_up, q_down, q_alpha, k_up, k_down, k_alpha,
                       v_up, v_down, v_alpha, mlp_up, mlp_down, mlp_alpha):
    """Join separate Q/K/V/MLP adapters back into a natively-fused QKVM adapter.

    Requires identical lora_down and alpha across all components — guaranteed after
    lora_split_qkv_mlp.
    """
    downs = [q_down, k_down, v_down, mlp_down]
    if not all(torch.equal(downs[0], d) for d in downs[1:]):
        raise ValueError("lora_join_qkv_mlp requires identical lora_down for all components")
    if not (q_alpha == k_alpha == v_alpha == mlp_alpha):
        raise ValueError("lora_join_qkv_mlp requires identical alpha for all components")
    return torch.cat([q_up, k_up, v_up, mlp_up], dim=0), q_down, q_alpha


def swap_chunks(input: torch.Tensor, dim: int=0) -> torch.Tensor:
    chunks = input.chunk(2, dim=dim)
    return torch.cat([chunks[1], chunks[0]], dim=dim)

def _not_implemented(*_args):
    raise NotImplementedError

def lora_qkv_fusion(q: str, k: str, v: str, qkv: str):
    return [
        ([f"{q}.lora_up.weight", f"{q}.lora_down.weight", f"{q}.alpha",
          f"{k}.lora_up.weight", f"{k}.lora_down.weight", f"{k}.alpha",
          f"{v}.lora_up.weight", f"{v}.lora_down.weight", f"{v}.alpha"],
         [f"{qkv}.lora_up.weight", f"{qkv}.lora_down.weight", f"{qkv}.alpha"],
         lora_fuse_qkv, lora_unfuse_qkv),
    ]

def lora_qkv_mlp_fusion(q: str, k: str, v: str, mlp: str, qkv_mlp: str, separator: str='.'):
    return [
        ([f"{q}.lora_up.weight",   f"{q}.lora_down.weight", f"{q}.alpha",
          f"{k}.lora_up.weight",   f"{k}.lora_down.weight", f"{k}.alpha",
          f"{v}.lora_up.weight",   f"{v}.lora_down.weight", f"{v}.alpha",
          f"{mlp}.lora_up.weight", f"{mlp}.lora_down.weight", f"{mlp}.alpha"],
         [f"{qkv_mlp}.lora_up.weight", f"{qkv_mlp}.lora_down.weight", f"{qkv_mlp}.alpha"],
         lora_fuse_qkv_mlp, lora_unfuse_qkv_mlp,
        ),

        #qkv only, in case there are no mlp layers:
        ([f"{q}.lora_up.weight",   f"{q}.lora_down.weight", f"{q}.alpha",
          f"{k}.lora_up.weight",   f"{k}.lora_down.weight", f"{k}.alpha",
          f"{v}.lora_up.weight",   f"{v}.lora_down.weight", f"{v}.alpha"],
         [f"{qkv_mlp}.lora_up.weight", f"{qkv_mlp}.lora_down.weight", f"{qkv_mlp}.alpha"],
          lambda q_up, q_down, q_alpha, k_up, k_down, k_alpha, v_up, v_down, v_alpha: lora_fuse_qkv_to_qkv_mlp(q_up, q_down, q_alpha, k_up, k_down, k_alpha, v_up, v_down, v_alpha),
          _not_implemented,
        ),

        #mlp only, in case there are no qkv layers:
        ([f"{mlp}.lora_up.weight", f"{mlp}.lora_down.weight", f"{mlp}.alpha"],
         [f"{qkv_mlp}.lora_up.weight", f"{qkv_mlp}.lora_down.weight", f"{qkv_mlp}.alpha"],
          lambda mlp_up, mlp_down, mlp_alpha: lora_fuse_mlp_to_qkv_mlp(mlp_up, mlp_down, mlp_alpha),
          _not_implemented,
        ),
    ]

def lora_qkv_native_split(qkv: str, q: str, k: str, v: str):
    """Pattern list: split a natively-fused QKV adapter into separate Q, K, V adapters.

    Use when loading ComfyUI LoRAs that have a single fused QKV adapter (lora_up shape
    (3*dim, rank)) rather than OT's block-diagonal adapter (shape (3*dim, 3*rank)).
    Distinguish the two cases by comparing lora_up.shape[1] against lora_down.shape[0].
    """
    return [
        ([f"{qkv}.lora_up.weight", f"{qkv}.lora_down.weight", f"{qkv}.alpha"],
         [f"{q}.lora_up.weight",   f"{q}.lora_down.weight",   f"{q}.alpha",
          f"{k}.lora_up.weight",   f"{k}.lora_down.weight",   f"{k}.alpha",
          f"{v}.lora_up.weight",   f"{v}.lora_down.weight",   f"{v}.alpha"],
         lora_split_qkv, lora_join_qkv),
    ]

def lora_qkv_mlp_native_split(qkvm: str, q: str, k: str, v: str, mlp: str):
    """Pattern list: split a natively-fused QKVM adapter into separate Q, K, V, MLP adapters.

    Assumes equal output dimensions for all four components. See lora_split_qkv_mlp for
    the equal-dim assumption and guidance on models with unequal dims.
    """
    return [
        ([f"{qkvm}.lora_up.weight", f"{qkvm}.lora_down.weight", f"{qkvm}.alpha"],
         [f"{q}.lora_up.weight",    f"{q}.lora_down.weight",    f"{q}.alpha",
          f"{k}.lora_up.weight",    f"{k}.lora_down.weight",    f"{k}.alpha",
          f"{v}.lora_up.weight",    f"{v}.lora_down.weight",    f"{v}.alpha",
          f"{mlp}.lora_up.weight",  f"{mlp}.lora_down.weight",  f"{mlp}.alpha"],
         lora_split_qkv_mlp, lora_join_qkv_mlp),
    ]

def qkv_fusion(q: str, k: str, v: str, qkv: str, separator: str='.'):
    return [
        ([q, k, v], qkv, fuse_qkv)
    ]

def qkv_mlp_fusion(q: str, k: str, v: str, mlp: str, qkv: str, separator: str='.'):
    return [
        ([q, k, v, mlp], qkv, fuse_qkv_mlp)
    ]
