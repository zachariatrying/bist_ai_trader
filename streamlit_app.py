import sys
import os

# Add src folder to python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Execute main streamlit application
app_path = os.path.join(src_dir, 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    code = compile(f.read(), app_path, 'exec')
    exec(code)
