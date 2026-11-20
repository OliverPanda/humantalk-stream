"""Bootstrap entry point; requires only Python 3.10+, no application dependencies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from krvoiceai.core.installation import main

if __name__ == "__main__":
    raise SystemExit(main())
