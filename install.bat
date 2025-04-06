@echo off
echo ===================================
echo AI Trading Bot Installer for Windows
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install required packages
echo Installing required packages...
pip install --upgrade pip
pip install ta pandas numpy matplotlib python-dotenv ccxt lightgbm joblib scikit-learn openai requests

REM Install FreqTrade
echo Installing FreqTrade...
pip install freqtrade

REM Create necessary directories
echo Creating directories...
mkdir user_data 2>nul
mkdir user_data\strategies 2>nul
mkdir user_data\models 2>nul
mkdir user_data\data 2>nul
mkdir user_data\logs 2>nul
mkdir user_data\backups 2>nul
mkdir config 2>nul
mkdir logs 2>nul

REM Modify .env file for AMD GPU support
echo Setting up GPU support for AMD RX6600...
echo # GPU Settings >> .env
echo GPU_PLATFORM_ID=0 >> .env
echo GPU_DEVICE_ID=0 >> .env

REM Create run.bat file
echo @echo off > run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo echo Starting FreqTrade AI Trading Bot >> run.bat
echo echo Select an option: >> run.bat
echo echo 1. Train AI models >> run.bat
echo echo 2. Run backtest >> run.bat
echo echo 3. Run trading bot >> run.bat
echo echo 4. Run simulation >> run.bat
echo echo 5. Exit >> run.bat
echo echo. >> run.bat
echo set /p option="Enter option (1-5): " >> run.bat
echo if "%%option%%"=="1" goto train >> run.bat
echo if "%%option%%"=="2" goto backtest >> run.bat
echo if "%%option%%"=="3" goto trade >> run.bat
echo if "%%option%%"=="4" goto simulation >> run.bat
echo if "%%option%%"=="5" goto end >> run.bat
echo goto invalid >> run.bat
echo. >> run.bat
echo :train >> run.bat
echo python run_freqtrade.py --mode train >> run.bat
echo goto end >> run.bat
echo. >> run.bat
echo :backtest >> run.bat
echo python run_freqtrade.py --mode backtest >> run.bat
echo goto end >> run.bat
echo. >> run.bat
echo :trade >> run.bat
echo python run_freqtrade.py --mode trade >> run.bat
echo goto end >> run.bat
echo. >> run.bat
echo :simulation >> run.bat
echo python simulation.py >> run.bat
echo goto end >> run.bat
echo. >> run.bat
echo :invalid >> run.bat
echo echo Invalid option. Please try again. >> run.bat
echo. >> run.bat
echo :end >> run.bat
echo pause >> run.bat

echo Installation complete! Run the bot by executing run.bat
echo.
echo IMPORTANT: Edit the .env file to add your API keys before running the bot.
echo.
pause