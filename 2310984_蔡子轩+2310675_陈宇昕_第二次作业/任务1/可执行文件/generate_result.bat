@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "TASK_DIR=%SCRIPT_DIR%.."
set "PROJECT_DIR="
set "RESULT_DIR="

for /d %%D in ("%TASK_DIR%\*") do (
    if exist "%%~fD\recommendation_project\src\predict_final.py" set "PROJECT_DIR=%%~fD\recommendation_project"
    if exist "%%~fD\final_model_metrics.csv" set "RESULT_DIR=%%~fD"
)

if not defined RESULT_DIR (
    for /d %%D in ("%TASK_DIR%\*") do (
        if exist "%%~fD\final_result.txt" set "RESULT_DIR=%%~fD"
    )
)

if not defined PROJECT_DIR (
    echo Failed to locate recommendation_project.
    goto end
)

if not defined RESULT_DIR (
    echo Failed to locate result directory.
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

echo Generating final prediction result...
"%PYTHON_CMD%" src\predict_final.py --model optimized_ensemble
if errorlevel 1 (
    echo Failed to generate prediction result.
    echo If numpy is missing, run setup_env.bat first.
    goto end
)

copy /Y "results\final_result.txt" "%RESULT_DIR%\final_result.txt"
if errorlevel 1 (
    echo Failed to copy final_result.txt.
    goto end
)

echo Generated: %RESULT_DIR%\final_result.txt
echo Done.

:end
if not defined NO_PAUSE pause
