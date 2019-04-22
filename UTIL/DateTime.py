import calendar
import datetime
import numpy as np


def get_dates_weekday(start_date, end_date):
    dt = end_date - start_date
    dates = []

    for i in range(dt.days + 1):
        date = start_date + datetime.timedelta(i)
        if calendar.weekday(date.year, date.month, date.day) < 5:
            dates.append(date)

    return dates


def get_time_section(timestamp, section_length):
    date_array   = np.array([int(t[8:10]) for t in timestamp])
    ind_new_date = np.insert(np.diff(date_array), 0, 0)
    section_idx  = 0
    section      = np.zeros_like(date_array)

    for i in range(len(section)):
        if ind_new_date[i] == 1:
            section_idx = 0
        if i % section_length == 0:
            section_idx += 1
        section[i] = section_idx

    return section
