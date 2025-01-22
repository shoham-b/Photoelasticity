import contextlib

import matlab.engine


@contextlib.contextmanager
def start_matlab():
    eng = matlab.engine.start_matlab()
    matlab_folder = r"C:\Users\shoha\PycharmProjects\PEGS"
    eng.cd(matlab_folder, nargout=0)
    yield eng
    eng.exit()
