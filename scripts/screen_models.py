#!/usr/bin/env python3
"""Back-compat entrypoint — prefer ``python -m scripts.screen`` / ``make screen``."""

import sys

from scripts.screen import main

if __name__ == "__main__":
    sys.exit(main())
