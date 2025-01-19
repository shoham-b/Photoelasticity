import contextlib


@contextlib.contextmanager
def start_malab():
    eng = matlab.engine.start_matlab()
    matlab_folder = rf"{__file__}\..\..\matlab"
    eng.cd(matlab_folder, nargout=0)
    yield eng
    eng.exit()
