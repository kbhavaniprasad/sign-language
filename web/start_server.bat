@echo off
echo ============================================================
echo   Sign Language Recognition - Web Interface Launcher
echo ============================================================
echo.
echo Starting backend API server...
echo The web interface will open automatically in your browser!
echo.
echo IMPORTANT: Keep this window open while using the interface!
echo.
echo ============================================================
echo.

cd /d d:\sign
python web\api_server.py

pause
