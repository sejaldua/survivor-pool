"""Simple entrypoint to run the solver from the repo root.
Usage: python scripts/run_solver.py
This will run the solver in src/ and write output to the root or data/ directory.
"""
import runpy
import os

HERE = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(HERE, 'src')
# run the solver module
runpy.run_path(os.path.join(SRC, 'solver.py'), run_name='__main__')
print('Solver executed')
