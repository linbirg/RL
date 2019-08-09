import time

def date_time_str_2_float(date_time_str):
    timeTuple = time.strptime(date_time_str, '%Y.%m.%d %H:%M')
    return time.mktime(timeTuple)