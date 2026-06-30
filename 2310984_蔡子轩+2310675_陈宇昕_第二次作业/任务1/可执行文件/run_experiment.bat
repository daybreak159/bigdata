@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "TASK_DIR=%SCRIPT_DIR%.."
set "PROJECT_DIR="

for /d %%D in ("%TASK_DIR%\*") do (
    if exist "%%~fD\recommendation_project\src\run_experiment.py" set "PROJECT_DIR=%%~fD\recommendation_project"
)

if not defined PROJECT_DIR (
    echo Failed to locate recommendation_project.
    goto end
)

set "VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" set "VENV_PYTHON=%PROJECT_DIR%\.venv\bin\python.exe"
set "PYTHON_CMD=python"
if exist "%VENV_PYTHON%" set "PYTHON_CMD=%VENV_PYTHON%"

echo Project directory:
echo %PROJECT_DIR%
echo.
echo Python command:
echo %PYTHON_CMD%
echo.

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo Failed to enter project directory.
    goto end
)

echo Running 5-fold experiment. This may take several minutes...
"%PYTHON_CMD%" src\run_experiment.py
if errorlevel 1 (
    echo Failed to run experiment.
    echo If numpy is missing, run setup_env.bat first.
    goto end
)

echo Done.

:end
if not defined NO_PAUSE pause
