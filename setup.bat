python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

REM Get the directory of the current script
set SCRIPT_DIR=%~dp0

pip install -r "%SCRIPT_DIR%requirements.txt"
echo Setup complete!
pause

start run-segtag.bat
exit

