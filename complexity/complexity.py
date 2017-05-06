import timeit
import numpy as np
from sklearn import linear_model
from multiprocessing import Process, Pipe


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def get_model(setup, statement, cleanup, sample_range):
    samples = np.arange(sample_range['min'], sample_range['max'], sample_range['step'])
    data_set = [setup(sample) for sample in samples]
    partially_applied = map(lambda data: wrapper(statement, data), data_set)
    timers = [timeit.Timer(stmt=fun) for fun in partially_applied]
    times = list(map(lambda x: x.timeit(10), timers))

    log_x = np.log(samples)
    log_y = np.log(times)
    model = linear_model.LinearRegression()
    model.fit(log_x.reshape(-1, 1), log_y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return {
        'degree': slope,
        'coefficient': np.exp(intercept)
    }


def __approximate(pipe_connection, setup, statement, cleanup):
    sample_range = {
        'min': 10,
        'max': 50,
        'step': 1
    }
    model = {
        'degree': None,
        'coefficient:': None
    }
    pipe_connection.send(model)

    while True:
        model = get_model(setup, statement, cleanup, sample_range)
        pipe_connection.send(model)
        sample_range['max'] += 10
        if sample_range['max'] >= 2000:
            sample_range['step'] = 100
            sample_range['max'] += 100
        elif sample_range['max'] >= 1000:
            sample_range['step'] = 50
            sample_range['max'] += 50
        elif sample_range['max'] >= 100:
            sample_range['step'] = 10
            sample_range['max'] += 10
        else:
            sample_range['max'] += 10


def approximate(setup, statement, cleanup, time=30):
    parent_connection, child_connection = Pipe()
    p = Process(target=__approximate, args=(child_connection,) + (setup, statement, cleanup))
    p.start()
    p.join(time)

    if p.is_alive():
        p.terminate()

    result = parent_connection.recv()
    while parent_connection.poll():
        result = parent_connection.recv()
    return result
