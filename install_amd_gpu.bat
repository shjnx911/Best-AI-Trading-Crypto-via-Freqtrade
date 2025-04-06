@echo off
echo ===================================
echo AMD RX6600 GPU Setup for TradingBot
echo ===================================
echo.

REM Create ROCm directory
mkdir AMD_GPU_Setup 2>nul
cd AMD_GPU_Setup

REM Download AMD drivers if not already present
if not exist amdgpu-install.exe (
    echo Downloading AMD ROCm installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.radeon.com/amdgpu-install/latest/windows/amdgpu-install-22.40.3.exe' -OutFile 'amdgpu-install.exe'"
)

REM Create Python script to configure LightGBM for GPU
echo Generating GPU configuration script...
echo import os > configure_lightgbm_gpu.py
echo import sys >> configure_lightgbm_gpu.py
echo. >> configure_lightgbm_gpu.py
echo # Check if AMD GPU is detected >> configure_lightgbm_gpu.py
echo try: >> configure_lightgbm_gpu.py
echo     import subprocess >> configure_lightgbm_gpu.py
echo     result = subprocess.run(['clinfo'], capture_output=True, text=True) >> configure_lightgbm_gpu.py
echo     devices = [line for line in result.stdout.split('\n') if 'Device Name' in line] >> configure_lightgbm_gpu.py
echo     amd_devices = [dev for dev in devices if 'AMD' in dev] >> configure_lightgbm_gpu.py
echo     rx6600_devices = [dev for dev in amd_devices if 'RX 6600' in dev] >> configure_lightgbm_gpu.py
echo. >> configure_lightgbm_gpu.py
echo     if rx6600_devices: >> configure_lightgbm_gpu.py
echo         print(f"AMD RX6600 GPU detected: {rx6600_devices[0]}") >> configure_lightgbm_gpu.py
echo         print("Configuring LightGBM for GPU acceleration...") >> configure_lightgbm_gpu.py
echo         # Get the environment variable file path from command line args >> configure_lightgbm_gpu.py
echo         env_file = sys.argv[1] if len(sys.argv) > 1 else ".env" >> configure_lightgbm_gpu.py
echo         with open(env_file, 'a') as f: >> configure_lightgbm_gpu.py
echo             f.write("\n# GPU Configuration for LightGBM\n") >> configure_lightgbm_gpu.py
echo             f.write("GPU_PLATFORM_ID=0\n") >> configure_lightgbm_gpu.py
echo             f.write("GPU_DEVICE_ID=0\n") >> configure_lightgbm_gpu.py
echo             f.write("LIGHTGBM_GPU_MODE=OpenCL\n") >> configure_lightgbm_gpu.py
echo         print(f"GPU configuration added to {env_file}") >> configure_lightgbm_gpu.py
echo. >> configure_lightgbm_gpu.py
echo         # Create a test script to verify GPU usage in LightGBM >> configure_lightgbm_gpu.py
echo         with open("test_gpu_lightgbm.py", 'w') as f: >> configure_lightgbm_gpu.py
echo             f.write("import lightgbm as lgb\n") >> configure_lightgbm_gpu.py
echo             f.write("import numpy as np\n") >> configure_lightgbm_gpu.py
echo             f.write("from sklearn.datasets import make_classification\n\n") >> configure_lightgbm_gpu.py
echo             f.write("# Generate sample data\n") >> configure_lightgbm_gpu.py
echo             f.write("X, y = make_classification(n_samples=10000, n_features=20)\n\n") >> configure_lightgbm_gpu.py
echo             f.write("# GPU parameters\n") >> configure_lightgbm_gpu.py
echo             f.write("params = {\n") >> configure_lightgbm_gpu.py
echo             f.write("    'boosting_type': 'gbdt',\n") >> configure_lightgbm_gpu.py
echo             f.write("    'objective': 'binary',\n") >> configure_lightgbm_gpu.py
echo             f.write("    'metric': 'binary_logloss',\n") >> configure_lightgbm_gpu.py
echo             f.write("    'device': 'gpu',\n") >> configure_lightgbm_gpu.py
echo             f.write("    'gpu_platform_id': 0,\n") >> configure_lightgbm_gpu.py
echo             f.write("    'gpu_device_id': 0\n") >> configure_lightgbm_gpu.py
echo             f.write("}\n\n") >> configure_lightgbm_gpu.py
echo             f.write("# Training with GPU\n") >> configure_lightgbm_gpu.py
echo             f.write("print('Training LightGBM model on GPU...')\n") >> configure_lightgbm_gpu.py
echo             f.write("lgb_train = lgb.Dataset(X, y)\n") >> configure_lightgbm_gpu.py
echo             f.write("gbm = lgb.train(params, lgb_train, num_boost_round=100, verbose_eval=20)\n") >> configure_lightgbm_gpu.py
echo             f.write("print('GPU Training completed successfully!')\n") >> configure_lightgbm_gpu.py
echo         print("Created GPU test script: test_gpu_lightgbm.py") >> configure_lightgbm_gpu.py
echo     else: >> configure_lightgbm_gpu.py
echo         print("No AMD RX6600 GPU detected. Make sure drivers are installed correctly.") >> configure_lightgbm_gpu.py
echo except Exception as e: >> configure_lightgbm_gpu.py
echo     print(f"Error detecting GPU: {e}") >> configure_lightgbm_gpu.py
echo     print("Please ensure AMD drivers are installed and OpenCL is properly configured.") >> configure_lightgbm_gpu.py

REM Create instructions file
echo Creating installation instructions...
echo === AMD RX6600 GPU Setup Instructions === > gpu_setup_instructions.txt
echo. >> gpu_setup_instructions.txt
echo 1. Install AMD GPU Drivers >> gpu_setup_instructions.txt
echo    - Run amdgpu-install.exe and follow the installation wizard >> gpu_setup_instructions.txt
echo    - Restart your computer after installation >> gpu_setup_instructions.txt
echo. >> gpu_setup_instructions.txt
echo 2. Install OpenCL Support >> gpu_setup_instructions.txt
echo    - Open AMD Radeon Software >> gpu_setup_instructions.txt
echo    - Go to Settings > System >> gpu_setup_instructions.txt
echo    - Ensure "OpenCL" is enabled >> gpu_setup_instructions.txt
echo. >> gpu_setup_instructions.txt
echo 3. Configure LightGBM for GPU acceleration >> gpu_setup_instructions.txt
echo    - After installing drivers and Python environment, run: >> gpu_setup_instructions.txt
echo      python configure_lightgbm_gpu.py >> gpu_setup_instructions.txt
echo. >> gpu_setup_instructions.txt
echo 4. Test GPU acceleration: >> gpu_setup_instructions.txt
echo    - Run: python test_gpu_lightgbm.py >> gpu_setup_instructions.txt
echo    - If successful, you'll see "GPU Training completed successfully!" >> gpu_setup_instructions.txt
echo. >> gpu_setup_instructions.txt
echo Note: If you encounter any issues, please refer to the AMD ROCm documentation: >> gpu_setup_instructions.txt
echo https://rocmdocs.amd.com/en/latest/ >> gpu_setup_instructions.txt

echo.
echo AMD GPU setup files created in the AMD_GPU_Setup directory.
echo Please follow the instructions in gpu_setup_instructions.txt to complete the setup.
echo.
cd ..
pause
