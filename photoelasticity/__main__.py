from photoelasticity.days.day3 import do_day_3, regenerate_day3_cache
from photoelasticity.days.day4 import do_day_4, regenerate_day4_cache

use_cache = True


def regenerate_all_cache():
    # regenerate_day3_cache()
    regenerate_day4_cache()
    return


def do_days():
    do_day_3()
    do_day_4()
    return


if __name__ == '__main__':
    regenerate_all_cache()
