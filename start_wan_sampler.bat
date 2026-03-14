@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
cd ..
python wan_sampler_gui.py
pause
