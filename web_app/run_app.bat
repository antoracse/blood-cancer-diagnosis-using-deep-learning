@echo off
REM Navigate to the web_app folder
cd /d "G:\blood-cancer-classification\web_app"

REM Run Streamlit with the venv Python without activating
..\venv\Scripts\python.exe -m streamlit run app.py

pause
