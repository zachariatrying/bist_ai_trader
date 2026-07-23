import sys
import os

# Add src folder to sys.path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import and launch main Streamlit application
import app
