import pathlib


def get_day_data(day: int, only=None):
    day_path = pathlib.Path(__file__).parent.parent / "data" / f"day{day}"
    if only is not None:
        return [day_path.joinpath(o) for o in only]
    all_files = list(day_path.glob("*"))
    return all_files
