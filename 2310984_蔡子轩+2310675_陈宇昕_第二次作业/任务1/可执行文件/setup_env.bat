@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "TASK_DIR=%SCRIPT_DIR%.."
set "PROJECT_DIR="

for /d %%D in ("%TASK_DIR%\*") do (
    if exist "%%~fD\recommendation_project\requirements.txt" set "PROJECT_DIR=%%~fD\recommendation_project"
)

if not defined PROJECT_DIR (
    echo Failed to locate recommendation_project.
    goto end
)

set "VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "ALT_VENV_PYTHON=%PROJECT_DIR%\.venv\bin\python.exe"

echo Project directory:
echo %PROJECT_DIR%
echo.

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo Failed to enter project directory.
    goto end
)

if not exist "%VENV_PYTHON%" if not exist "%ALT_VENV_PYTHON%" (
    echo Creating local virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Please check whether Windows Python is installed and available in PATH.
        goto end
    )
)

if not exist "%VENV_PYTHON%" set "VENV_PYTHON=%ALT_VENV_PYTHON%"
if not exist "%VENV_PYTHON%" (
    echo Failed to locate virtual environment Python.
    goto end
)

echo Virtual environment Python:
echo %VENV_PYTHON%
echo.

echo Installing dependencies from requirements.txt...
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    goto end
)

echo Checking numpy...
"%VENV_PYTHON%" -c "import numpy; print('numpy', numpy.__version__)"
if errorlevel 1 (
    echo numpy check failed.
    goto end
)

echo Environment is ready.
echo You can now run generate_result.bat.

:end
if not defined NO_PAUSE pause
