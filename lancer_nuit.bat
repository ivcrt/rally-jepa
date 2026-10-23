@echo off
REM Collecte sur terre pendant la nuit, puis extinction du PC.
REM Double-clique sur ce fichier (il doit etre a la racine de rally-jepa).
cd /d "%~dp0"
if not exist outputs mkdir outputs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM Pas de mise en veille sur secteur (sinon la collecte s'arrete).
powercfg /change standby-timeout-ac 0

REM Ferme le jeu s'il est encore ouvert (le script lance sa propre instance).
taskkill /F /IM BeamNG.tech.x64.exe >nul 2>&1
taskkill /F /IM BeamNG.tech.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo Collecte en cours... le journal est dans outputs\collect_dirt_log.txt
echo Debut : %date% %time% > outputs\collect_dirt_log.txt
python data\collect_dirt.py >> outputs\collect_dirt_log.txt 2>&1
echo Fin : %date% %time% >> outputs\collect_dirt_log.txt

REM Extinction dans 2 minutes. Pour annuler : shutdown /a
shutdown /s /t 120
