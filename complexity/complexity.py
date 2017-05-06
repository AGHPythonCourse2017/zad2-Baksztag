import logging
import timeit
from multiprocessing import Process, Pipe

import numpy as np
from sklearn import linear_model


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def get_model(setup, statement, cleanup, sample_range):
    samples = np.arange(sample_range['min'], sample_range['max'], sample_range['step'])
    data_set = [setup(sample) for sample in samples]
    partially_applied = map(lambda data: wrapper(statement, data), data_set)
    timers = [timeit.Timer(stmt=fun) for fun in partially_applied]
    times = list(map(lambda x: x.timeit(10) / 10, timers))

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
    logging.basicConfig(level=logging.DEBUG)
    sample_range = {
        'min': 10,
        'max': 50,
        'step': 1
    }
    model = {
        'degree': 0.0,
        'coefficient:': 0.0
    }
    pipe_connection.send(model)

    while True:
        logging.info('Approximating in data range {min: %d, max: %d, step: %d}.',
                     sample_range['min'],
                     sample_range['max'],
                     sample_range['step']
                     )
        model = get_model(setup, statement, cleanup, sample_range)
        if abs(model['degree']) < 1e-1:
            sample_range['step'] *= 2
            sample_range['max'] *= 2
            sample_range['min'] *= 2
        pipe_connection.send(model)
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
        logging.info('Current coefficient: %f', model['degree'])


def __get_complexity(degree):
    if degree < 0.5:
        return "O(1)"
    elif 0.5 <= degree and degree < 1.05:
        return "O(n)"
    elif 1.05 <= degree and degree < 1.1:
        return "O(n) or O(nlogn)"
    elif 1.1 <= degree and degree < 1.5:
        return "O(nlogn)"
    elif 1.5 <= degree and degree < 2.5:
        return "O(n^2)"
    elif 2.5 <= degree and degree < 3.5:
        return "O(n^3)"
    elif 3.5 <= degree and degree < 4.5:
        return "O(n^4)"
    else:
        return "Exponential complexity"


def approximate(setup, statement, cleanup, timeout=30):
    parent_connection, child_connection = Pipe()
    p = Process(target=__approximate, args=(child_connection,) + (setup, statement, cleanup))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()

    result = parent_connection.recv()
    while parent_connection.poll():
        result = parent_connection.recv()

    complexity = __get_complexity(result['degree'])

    def get_time(n):
        return result['coefficient'] * n ** result['degree']

    def get_size(t):
        return (t / result['coefficient']) ** (1 / result['degree'])

    return {
        'complexity': complexity,
        'time_model': get_time,
        'size_model': get_size
    }
