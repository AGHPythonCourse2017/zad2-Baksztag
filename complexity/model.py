import timeit
import numpy as np
from sklearn import linear_model


def get_model(setup, statement, cleanup):
    samples = np.arange(100, 5000, 100)
    print(samples)
    data_set = list(map(setup, samples))
    print(data_set)
    timers = [timeit.Timer(lambda: statement(data)) for data in data_set]
    times = list(map(lambda x: x.timeit(1), timers))
    print(times)

    log_x = np.log(samples)
    log_y = np.log(times)
    model = linear_model.LinearRegression()
    model.fit(log_x.reshape(-1, 1), log_y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return [slope, np.exp(intercept)]
