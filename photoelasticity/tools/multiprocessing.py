import multiprocessing
from os import cpu_count


def with_pool():
    return multiprocessing.Pool(cpu_count() // 2)
