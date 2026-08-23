@echo off
rem ===================================================================
rem  Deniz Yildizi Asistani - masaustu baslatici
rem
rem  Cift tiklayin. Sirasiyla:
rem    1) yerel sunucuyu baslatir (site\sunucu.py)
rem    2) model yuklenene kadar bekler (/saglik yoklamasi)
rem    3) sayfayi Edge'in uygulama kipinde acar: adres cubugu ve sekme yok
rem
rem  Kapatmak icin: acilan pencereyi kapatin, sonra kucultulmus
rem  "Deniz Yildizi - sunucu" penceresini kapatin (ya da orada Ctrl+C).
rem ===================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "ADRES=http://127.0.0.1:8000"
set "YOL=/uygulama"

rem --- Python: once proje venv'i, yoksa sistemdeki python ---
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

rem --- Sunucu zaten calisiyorsa yeniden baslatma ---
call :saglik
if "!DURUM!"=="1" (
  echo Sunucu zaten calisiyor.
) else (
  echo Sunucu baslatiliyor, model yukleniyor...
  echo Bu ilk seferde 10-15 saniye surebilir.
  start "Deniz Yildizi - sunucu" /min "%PY%" "%~dp0site\sunucu.py"
)

rem --- Hazir olmasini bekle (en fazla 90 saniye) ---
set /a SAYAC=0
:bekle
call :saglik
if "!DURUM!"=="1" goto hazir
set /a SAYAC+=1
if !SAYAC! GEQ 90 goto zaman_asimi
timeout /t 1 /nobreak >nul
goto bekle

:hazir
echo Hazir. Uygulama aciliyor...

rem --- Uygulama kipinde ac: once Edge, sonra Chrome, olmazsa varsayilan ---
set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE%" set "EDGE=C:\Program Files\Microsoft\Edge\Application\msedge.exe"
set "CHROME=C:\Program Files\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

if exist "%EDGE%" (
  start "" "%EDGE%" --app=%ADRES%%YOL% --window-size=920,760
) else if exist "%CHROME%" (
  start "" "%CHROME%" --app=%ADRES%%YOL% --window-size=920,760
) else (
  echo Edge veya Chrome bulunamadi; varsayilan tarayicida aciliyor.
  start "" "%ADRES%%YOL%"
)
exit /b 0

:zaman_asimi
echo.
echo Sunucu 90 saniyede hazir olmadi.
echo Muhtemel sebep: Foundry Local kurulu degil ya da servisi calismiyor.
echo Kontrol icin:  foundry service status
echo Ayrintili hata icin kucultulmus "Deniz Yildizi - sunucu" penceresine bakin.
echo.
pause
exit /b 1

rem --- /saglik yoklamasi: DURUM=1 ise sunucu ayakta ---
:saglik
set "DURUM=0"
powershell -NoProfile -Command ^
  "try { $r = Invoke-WebRequest -Uri '%ADRES%/saglik' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 set "DURUM=1"
exit /b 0
