"""Запуск программы (этот файл собирается в exe через PyInstaller)."""
import sys

from wplace_bot.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
