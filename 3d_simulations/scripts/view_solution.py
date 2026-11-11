"""View a solved 3D problem in the browser. Thin wrapper around ``biosym-view``:

    python scripts/view_solution.py RESULT.pkl [--cycles 3] [--port 8080] [--geometry DIR]
"""
from biosym.results.viewer3d import main

if __name__ == "__main__":
    main()
