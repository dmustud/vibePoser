@echo off
set "BASE_DIR=%~dp0"
set "PIXI_INTERPRETER=%BASE_DIR%.pixi\envs\default\pythonw.exe"

echo Checking Vibe Poser Environment at: %BASE_DIR%

if exist "%PIXI_INTERPRETER%" (
    echo Starting Vibe Poser with Pixi interpreter...
    pushd "%BASE_DIR%"
    start "" "%PIXI_INTERPRETER%" "%BASE_DIR%pose_app.py"
    popd
) else (
    echo [WARNING] Pixi environment not found at %PIXI_INTERPRETER%
    echo Attempting to run with system python...
    start "" pythonw "%BASE_DIR%pose_app.py"
)

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to start Vibe Poser.
    pause
)
exit