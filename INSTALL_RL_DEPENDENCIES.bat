@echo off
setlocal DisableDelayedExpansion
call "%~dp0START_URBANFLOW.bat" --install-dependencies --check-only
exit /b %errorlevel%
