@echo off
cd /d "%~dp0"

set PORT=8000

REM Kiem tra cong da bi chiem chua -> bao ro thay vi de uvicorn van loi kho hieu.
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo [LOI] Cong %PORT% dang bi mot chuong trinh khac chiem giu.
    echo       Thuong la do server nay da chay san o cua so khac,
    echo       hoac trinh xem truoc trong trinh soan thao dang giu cong.
    echo.
    echo   Cach xu ly:
    echo     1. Dong cua so terminal dang chay server cu, hoac
    echo     2. Xem tien trinh giu cong:  netstat -ano ^| findstr :%PORT%
    echo        roi tat no:              taskkill /PID ^<so_PID^> /F
    echo     3. Hoac doi cong: dat  set PORT=8010  trong file nay.
    echo.
    echo   Neu server da chay san, chi can mo trinh duyet tai:
    echo       http://localhost:%PORT%
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Khong tim thay moi truong ao .venv trong thu muc du an.
    echo       Hay tao va cai dat truoc:
    echo         python -m venv .venv
    echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Dang khoi dong Groq Assistant tai http://localhost:%PORT%  (nhan Ctrl+C de dung)
.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port %PORT%
