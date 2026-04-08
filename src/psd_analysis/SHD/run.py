from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.psd_analysis.run import main as generic_main


if __name__ == '__main__':
    if '--dataset' not in sys.argv:
        sys.argv = [sys.argv[0], '--dataset', 'shd', *sys.argv[1:]]
    generic_main()
