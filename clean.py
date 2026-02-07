#!/usr/bin/env python
"""
Cleanup script for the exovista repository.

Removes generated files and directories:
- *.exo files (excluding test/data)
- __pycache__ directories
- .pytest_cache directory
- Temporary test output files
"""

import os
import shutil
import glob
import logging

logging.basicConfig(level=logging.INFO)


def clean() -> None:
    """
    Remove generated files and directories from the repository.
    
    This function cleans up:
    - All .exo files except those in test/data (presumed source data)
    - All __pycache__ directories
    - .pytest_cache directory
    - Temporary test output files (test_output*.txt)
    """
    # Remove all .exo files in the root and subdirectories
    # But preserve test/data which contains source test files
    exo_files = glob.glob("**/*.exo", recursive=True)
    for f in exo_files:
        if "test/data" in f:
            logging.info(f"Skipping {f} (presumed test data)")
            continue
        try:
            os.remove(f)
            logging.info(f"Removed {f}")
        except Exception as e:
            logging.error(f"Error removing {f}: {e}")

    # Remove __pycache__ directories
    pycache_dirs = glob.glob("**/__pycache__", recursive=True)
    for d in pycache_dirs:
        try:
            shutil.rmtree(d)
            logging.info(f"Removed {d}")
        except Exception as e:
            logging.error(f"Error removing {d}: {e}")

    # Remove temporary test outputs if any
    temp_files = [
        "test_output.txt",
        "test_output_2d.txt",
        "test_output_2d_v2.txt",
        "test_output_write_exo.txt"
    ]
    for f in temp_files:
        if os.path.exists(f):
            try:
                os.remove(f)
                logging.info(f"Removed {f}")
            except Exception as e:
                logging.error(f"Error removing {f}: {e}")

    # Remove .pytest_cache
    if os.path.exists(".pytest_cache"):
        try:
            shutil.rmtree(".pytest_cache")
            logging.info("Removed .pytest_cache")
        except Exception as e:
            logging.error(f"Error removing .pytest_cache: {e}")


if __name__ == "__main__":
    clean()
