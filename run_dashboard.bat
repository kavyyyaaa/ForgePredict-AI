@echo off
title ForgePredict AI Dashboard Launcher
echo ==========================================================
echo   ForgePredict AI - F1 Predictive Maintenance System
echo ==========================================================
echo.

cd /d "%~dp0"

:: Check if virtual environment is valid (handles path changes when copying via pen drive)
if exist ".venv" (
    .venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Existing virtual environment is invalid (likely moved to a new path).
        echo Re-creating virtual environment for the new path...
        echo.
        rmdir /s /q .venv
    )
)

:: Check if virtual environment exists; if so, skip setup
if exist ".venv" goto :ACTIVATE

echo [NOTICE] Virtual environment (.venv) not found.
echo Setting up environment...
echo.

:: Check if uv is in PATH
where uv >nul 2>&1
if %errorlevel% equ 0 (
    set "USE_UV=1"
    set "RUN_CMD=uv"
    goto :CREATE_VENV
)

:: Check if uv is in default local bin
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "USE_UV=1"
    set "RUN_CMD=%USERPROFILE%\.local\bin\uv.exe"
    goto :CREATE_VENV
)

:: Try to install uv
echo Attempting to install 'uv' package manager for fast environment setup...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "USE_UV=1"
    set "RUN_CMD=%USERPROFILE%\.local\bin\uv.exe"
    goto :CREATE_VENV
)

:: Fallback to standard python venv
echo [NOTICE] 'uv' not found. Falling back to standard python venv...
set "USE_UV=0"

:CREATE_VENV
echo.
echo Creating virtual environment (.venv)...
if "%USE_UV%"=="1" (
    "%RUN_CMD%" venv
) else (
    python -m venv .venv
)

echo.
echo Installing dependencies from requirements.txt...
if "%USE_UV%"=="1" (
    "%RUN_CMD%" pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

echo.
echo Environment setup completed!
echo ==========================================================
echo.

:ACTIVATE
echo [1/2] Activating Python virtual environment...
call .venv\Scripts\activate.bat

echo [2/2] Launching local web server...
echo Server will be available at: http://127.0.0.1:8050/
echo.
echo (To stop the server, close this window or press Ctrl+C)
echo ==========================================================
echo.

:: Wait 3 seconds in background using ping, then open default browser
start /b cmd /c "ping -n 4 127.0.0.1 >nul && start http://127.0.0.1:8050/"

:: Start Dash app
python app.py

:: If it exits or crashes, show pause so the user can read the error
echo.
echo [ERROR] The dashboard server has stopped or failed to start.
pause
