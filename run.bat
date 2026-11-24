@echo off
REM Starts the RAG agent locally: sets up the venv on first run,
REM builds the knowledge base if it's missing, then opens the dashboard.
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv || goto :error
    echo Installing dependencies, this can take a few minutes...
    venv\Scripts\python -m pip install --upgrade pip || goto :error
    venv\Scripts\python -m pip install -r requirements.txt || goto :error
)

if not exist .env (
    echo No .env found, copying .env.example. Edit it to add your API key.
    copy .env.example .env >nul
)

if not exist data\chroma\chroma.sqlite3 (
    echo Building the knowledge base from sample_docs\ ...
    venv\Scripts\python ingest.py || goto :error
)

echo Starting the dashboard at http://localhost:8501
venv\Scripts\python -m streamlit run app.py
goto :eof

:error
echo.
echo Setup failed, see the error above.
pause
exit /b 1
