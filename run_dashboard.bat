@echo off
title ForgePredict AI Dashboard Launcher
echo ==========================================================
echo   ForgePredict AI - F1 Predictive Maintenance System
echo ==========================================================
echo.

cd /d "%~dp0"

:: Check if virtual environment exists
if not exist ".venv" (
    echo [NOTICE] Virtual environment (.venv) not found.
    echo Setting up environment on this new machine...
    echo.
    
    :: Check if uv is in PATH
    where uv >nul 2>&1
    if %errorlevel% neq 0 (
        :: Check if uv is in the default local bin folder
        if exist "%USERPROFILE%\.local\bin\uv.exe" (
            set "UV_PATH=%USERPROFILE%\.local\bin\uv.exe"
        ) else (
            echo Installing 'uv' package manager for fast environment setup...
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            set "UV_PATH=%USERPROFILE%\.local\bin\uv.exe"
        )
    ) else (
        set "UV_PATH=uv"
    )
    
    echo.
    echo Creating virtual environment (.venv)...
    "%UV_PATH%" venv
    
    echo.
    echo Installing dependencies from requirements.txt...
    "%UV_PATH%" pip install -r requirements.txt
    
    echo.
    echo Environment setup completed!
    echo ==========================================================
    echo.
)

echo [1/2] Activating Python virtual environment...
call .venv\Scripts\activate.bat

echo [2/2] Launching local web server...
echo Server will be available at: http://127.0.0.1:8050/
echo.
echo (To stop the server, close this window or press Ctrl+C)
echo ==========================================================
echo.

:: Wait 3 seconds in background to allow models to train, then open default browser
start /b cmd /c "timeout /t 3 >nul && start http://127.0.0.1:8050/"

:: Start Dash app
python app.py

pause
