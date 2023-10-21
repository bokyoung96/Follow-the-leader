"""
Article: Follow the leader: Index tracking with factor models

Topic: ETC
"""
import time


def time_spent_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        run = func(*args, **kwargs)
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        print(f"{func.__name__} EXECUTED TIME: {execution_time} SECONDS")
        return run
    return wrapper
