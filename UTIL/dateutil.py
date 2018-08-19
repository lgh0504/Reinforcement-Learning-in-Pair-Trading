import calendar
import datetime


def get_dates_weekday(start_date, end_date):
    dt = end_date - start_date
    dates = []
    for i in range(dt.days + 1):
        date = start_date + datetime.timedelta(i)
        if calendar.weekday(date.year, date.month, date.day) < 5:
            dates.append(date)
    return dates
