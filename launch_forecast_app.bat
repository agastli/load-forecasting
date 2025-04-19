@echo off
echo [⚙️  Initializing Environment...]

REM Set correct Python path - update if yours is installed elsewhere
set PYTHON_PATH=C:\Users\adelg\AppData\Local\Programs\Python\Python312\python.exe

IF NOT EXIST .venv (
    echo [📦 Creating virtual environment...]
    "%PYTHON_PATH%" -m venv .venv
)

echo [✅ Activating environment...]
call .venv\Scripts\activate.bat

echo [⬆️  Upgrading pip...]
pip install --upgrade pip

echo [📦 Installing required packages...]
pip install -r requirements.txt

echo [🚀 Launching Streamlit App...]
start "" http://localhost:8501
streamlit run app/streamlit_app.py
pause
