# simple_rag_project/conftest.py
import sys
import os

# Add the project root directory to the Python path.
# This allows pytest to find and import modules from the 'src' directory.
# os.path.dirname(__file__) gives the directory where conftest.py is located (project root).
# os.path.abspath ensures it's an absolute path.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))