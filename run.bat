@echo off
REM FreqTrade Bot Runner Script
REM Được tạo để quản lý các tác vụ khác nhau của FreqTrade AI Bot

REM Màu sắc cho menu
color 0A

:MENU
cls
echo ==============================================
echo     FREQTRADE AI TRADING BOT - MANAGER
echo ==============================================
echo.
echo  1. Train AI models
echo  2. Run backtest
echo  3. Run trading bot
echo  4. Run simulation
echo  5. Update pair ranking
echo  6. Setup config with wizard
echo  7. Check bot health
echo  8. Generate report
echo  9. Setup AMD GPU
echo  0. Exit
echo.
echo ==============================================
set /p choice=Enter your choice (0-9): 

if "%choice%"=="1" goto TRAIN
if "%choice%"=="2" goto BACKTEST
if "%choice%"=="3" goto TRADE
if "%choice%"=="4" goto SIMULATION
if "%choice%"=="5" goto PAIRRANKING
if "%choice%"=="6" goto SETUPCONFIG
if "%choice%"=="7" goto CHECKHEALTH
if "%choice%"=="8" goto REPORT
if "%choice%"=="9" goto SETUPGPU
if "%choice%"=="0" goto EXIT

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

:TRAIN
cls
echo ==============================================
echo            TRAINING AI MODELS
echo ==============================================
echo.
echo Training AI models for the trading strategy...
echo This may take several minutes depending on your hardware.
echo.
python run_freqtrade.py --mode train
echo.
echo Training completed.
echo.
pause
goto MENU

:BACKTEST
cls
echo ==============================================
echo             RUNNING BACKTEST
echo ==============================================
echo.
set /p days=Enter number of days to backtest (default: 30): 
if "%days%"=="" set days=30

set /p pairs=Enter pairs to backtest (comma separated, default: BTC/USDT,ETH/USDT,SOL/USDT): 
if "%pairs%"=="" set pairs=BTC/USDT,ETH/USDT,SOL/USDT

echo Running backtest for %days% days on pairs: %pairs%
echo.
python run_freqtrade.py --mode backtest --days %days% --pairs %pairs%
echo.
echo Backtest completed.
echo.
pause
goto MENU

:TRADE
cls
echo ==============================================
echo            STARTING TRADING BOT
echo ==============================================
echo.
echo IMPORTANT: Make sure you have configured your API keys
echo and reviewed your trading settings before starting.
echo.
set /p confirm=Are you sure you want to start trading? (y/n): 
if /i not "%confirm%"=="y" goto MENU

echo Starting trading bot...
echo.
start "FreqTrade Trading Bot" cmd /c "python run_freqtrade.py --mode trade && pause"
echo.
echo Bot has been started in a new window.
echo.
pause
goto MENU

:SIMULATION
cls
echo ==============================================
echo           RUNNING SIMULATION
echo ==============================================
echo.
echo Running trading simulation with mock data...
echo.
python simulation.py
echo.
echo Simulation completed.
echo.
pause
goto MENU

:PAIRRANKING
cls
echo ==============================================
echo          UPDATING PAIR RANKING
echo ==============================================
echo.
echo Finding the top 5 most profitable pairs...
echo.
python pair_ranking.py
echo.
echo Pair ranking updated.
echo Results saved to pair_ranking.json
echo New config created at config/top_pairs_config.json
echo.
pause
goto MENU

:SETUPCONFIG
cls
echo ==============================================
echo           SETUP CONFIGURATION
echo ==============================================
echo.
echo Running configuration wizard...
echo.
python setup_config.py --wizard
echo.
echo Configuration setup completed.
echo.
pause
goto MENU

:CHECKHEALTH
cls
echo ==============================================
echo            CHECKING BOT HEALTH
echo ==============================================
echo.
echo Checking trading bot health status...
echo.
python monitoring.py --check-health
echo.
pause
goto MENU

:REPORT
cls
echo ==============================================
echo           GENERATING REPORT
echo ==============================================
echo.
echo Generating trading performance report...
echo.
python monitoring.py --generate-report
echo.
pause
goto MENU

:SETUPGPU
cls
echo ==============================================
echo           SETUP AMD GPU SUPPORT
echo ==============================================
echo.
echo Setting up AMD RX6600 GPU acceleration...
echo.
call install_amd_gpu.bat
echo.
pause
goto MENU

:EXIT
cls
echo Thank you for using FreqTrade AI Trading Bot!
echo Exiting...
timeout /t 2 >nul
exit
