import contextlib

import matlab.engine


@contextlib.contextmanager
def start_matlab():
    eng = matlab.engine.start_matlab()
    matlab_folder = rf"{__file__}\..\..\matlab"
    eng.cd(matlab_folder, nargout=0)
    yield eng
    eng.exit()
