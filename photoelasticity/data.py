import pathlib


def get_day_data(day: int):
    day_path = pathlib.Path(__file__).parent.parent/"data"/f"day{day}"
    all_files = day_path.glob("*")
    return all_files