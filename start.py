"""FreeVideoAI 一键启动入口

用法：
    python start.py            # 启动 Web 界面（默认 http://127.0.0.1:8000）
    python start.py --port 9000
"""
import sys

from krvoiceai.ui.cli import main

if __name__ == "__main__":
    # 等价于 python -m krvoiceai.ui.cli web [参数]
    sys.argv = [sys.argv[0], "web", *sys.argv[1:]]
    sys.exit(main())
