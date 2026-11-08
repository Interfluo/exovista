"""``python -m exovista`` runs the ``exovista-read`` command."""

import sys

from .exoread import main

if __name__ == "__main__":
    sys.exit(main())
