import sys
from pathlib import Path

# Папка ноды содержит дефис в имени → как пакет не импортируется.
# Кладём её в sys.path и импортируем модули напрямую.
_NODE_DIR = str(Path(__file__).resolve().parents[1])
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)
