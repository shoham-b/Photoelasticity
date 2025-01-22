import contextlib
from pathlib import Path

import matlab.engine


@contextlib.contextmanager
def start_matlab():
    eng = matlab.engine.start_matlab()
    matlab_folder = (Path(__file__).parent.parent.parent / "PEGS").absolute()
    eng.cd(str(matlab_folder), nargout=0)
    yield eng
    eng.exit()
