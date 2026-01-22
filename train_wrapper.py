"""
Wrapper script for running torchgeo training on Windows

This wrapper adds the required multiprocessing guard for Windows
"""
import multiprocessing
import sys
from pathlib import Path

# Adding sys.path to help with finding the dataset
current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# adding the parent folder to the sys.path
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from torchgeo.main import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()