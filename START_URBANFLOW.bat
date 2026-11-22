@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
where python.exe >nul 2>nul
if errorlevel 1 goto no_python
python.exe -X utf8 -m urbanflow.launcher %*
exit /b %errorlevel%

:no_python
echo Python was not found on PATH. Activate your UrbanFlow Python environment and try again.
exit /b 1
