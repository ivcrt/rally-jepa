@echo off
REM Collecte avec camera : test automatique, puis collecte complete, puis extinction.
REM Double-clique sur ce fichier (il doit etre a la racine de rally-jepa).
cd /d "%~dp0"
if not exist outputs mkdir outputs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
powercfg /change standby-timeout-ac 0
set LOG=outputs\collect_cam_log.txt

taskkill /F /IM BeamNG.tech.x64.exe >nul 2>&1
taskkill /F /IM BeamNG.tech.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo Test de la camera (2 a 4 minutes)... journal : %LOG%
echo Debut : %date% %time% > %LOG%
python data\collect_cam.py --test >> %LOG% 2>&1
if errorlevel 1 goto echec

echo Test OK. Ouvre outputs\apercu_camera.png pour verifier l'image. Collecte complete en cours...
python data\collect_cam.py >> %LOG% 2>&1
echo Fin : %date% %time% >> %LOG%
goto fin

:echec
echo TEST CAMERA EN ECHEC, collecte annulee. >> %LOG%
echo Test camera en echec : voir %LOG%

:fin
REM Extinction dans 2 minutes. Pour annuler : shutdown /a
shutdown /s /t 120
