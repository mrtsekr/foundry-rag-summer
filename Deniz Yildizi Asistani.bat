@echo off
rem ===================================================================
rem  Deniz Yildizi Asistani - masaustu baslatici
rem
rem  ONEMLI: Bu dosya DEPO KLASORUNUN ICINDE durmalidir. Tek basina
rem  indirilip calistirilamaz; yanindaki site\sunucu.py'yi ve projeyi
rem  arar. Once depoyu klonlayin, kurulumu yapin, sonra buna cift tiklayin.
rem
rem  Sirasiyla:
rem    1) onkosullari kontrol eder (proje dosyalari, Python, bilgi tabani)
rem    2) yerel sunucuyu baslatir
rem    3) model yuklenene kadar bekler
rem    4) sayfayi Edge'in uygulama kipinde acar
rem ===================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Deniz Yildizi Asistani - baslatici

set "ADRES=http://127.0.0.1:8000"
set "YOL=/uygulama"

rem ---------- 1) ONKOSULLAR ----------
rem Eksik bir sey varsa 90 saniye beklemek yerine HEMEN soyluyoruz.

if not exist "%~dp0site\sunucu.py" (
  echo.
  echo   HATA: site\sunucu.py bulunamadi.
  echo.
  echo   Bu dosya depo klasorunun ICINDE calismalidir. Tek basina
  echo   indirdiyseniz calismaz. Once depoyu klonlayin:
  echo.
  echo     git clone https://github.com/mrtsekr/foundry-rag-summer
  echo.
  echo   Sonra bu dosyaya depo klasorunun icinden cift tiklayin.
  echo.
  pause
  exit /b 1
)

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  where python >nul 2>&1
  if errorlevel 1 (
    echo.
    echo   HATA: Python bulunamadi.
    echo   Python 3.12 kurun, sonra depo klasorunde:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
  )
  set "PY=python"
  echo   Not: .venv yok, sistem Python'u kullanilacak.
)

if not exist "%~dp0rag_store.db" (
  echo.
  echo   HATA: Bilgi tabani kurulmamis ^(rag_store.db yok^).
  echo   Once su komutu calistirin:
  echo.
  echo     python ingest.py
  echo.
  pause
  exit /b 1
)

rem curl.exe Windows 10+ ile geliyor ve powershell'den hizli aciliyor.
set "CURL=%SystemRoot%\System32\curl.exe"

rem ---------- 2) SUNUCU ----------
call :saglik
if "!DURUM!"=="1" (
  echo   Sunucu zaten calisiyor.
) else (
  echo   Sunucu baslatiliyor...
  start "Deniz Yildizi - sunucu" /min "%PY%" "%~dp0site\sunucu.py"
)

rem ---------- 3) BEKLE ----------
echo   Model yukleniyor. Ilk acilis 10-15 saniye surebilir.
set /a SAYAC=0
:bekle
call :saglik
if "!DURUM!"=="1" goto hazir
set /a SAYAC+=1
if !SAYAC! GEQ 60 goto zaman_asimi
rem Her 5 saniyede bir isaret: pencere donmus gibi gorunmesin.
set /a KALAN=SAYAC %% 5
if !KALAN!==0 echo     ... !SAYAC! saniye
timeout /t 1 /nobreak >nul
goto bekle

:hazir
echo   Hazir. Uygulama aciliyor...

set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE%" set "EDGE=C:\Program Files\Microsoft\Edge\Application\msedge.exe"
set "CHROME=C:\Program Files\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

if exist "%EDGE%" (
  start "" "%EDGE%" --app=%ADRES%%YOL% --window-size=920,760
) else if exist "%CHROME%" (
  start "" "%CHROME%" --app=%ADRES%%YOL% --window-size=920,760
) else (
  echo   Edge/Chrome bulunamadi, varsayilan tarayicida aciliyor.
  start "" "%ADRES%%YOL%"
)
exit /b 0

:zaman_asimi
echo.
echo   Sunucu 60 saniyede hazir olmadi.
echo.
echo   En olasi sebep: Foundry Local kurulu degil ya da servisi calismiyor.
echo   Kontrol:  foundry service status
echo.
echo   Gercek hatayi gormek icin gorev cubugundaki
echo   "Deniz Yildizi - sunucu" penceresini buyutun.
echo.
pause
exit /b 1

rem ---------- /saglik yoklamasi: DURUM=1 ise ayakta ----------
:saglik
set "DURUM=0"
if exist "%CURL%" (
  "%CURL%" -s -o nul -m 2 "%ADRES%/saglik"
  if not errorlevel 1 set "DURUM=1"
) else (
  powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%ADRES%/saglik' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
  if not errorlevel 1 set "DURUM=1"
)
exit /b 0
