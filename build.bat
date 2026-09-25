@echo off
rem Сборка WplaceBot.exe на Windows (нужен Python 3.10+ с галочкой "Add to PATH").
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name WplaceBot --hidden-import wplace_bot.gui run.py
echo.
echo Готово: dist\WplaceBot.exe
pause
