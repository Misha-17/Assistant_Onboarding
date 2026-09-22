@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Create .venv and install requirements.txt first. See README.md.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0run_assistant.py" %*
exit /b %errorlevel%
