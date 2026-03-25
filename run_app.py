import os
import sys

# Ensure project root is on sys.path so `import src...` works
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import chainlit.cli

if __name__ == "__main__":
    sys.argv = ["chainlit", "run", "src/app.py", "-w"]
    chainlit.cli.main()