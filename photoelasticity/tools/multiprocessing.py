import multiprocessing
from os import cpu_count


def with_pool():
    return multiprocessing.Pool(3 * cpu_count() // 4)
