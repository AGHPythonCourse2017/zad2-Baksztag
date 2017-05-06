from complexity import complexity


def setup(n):
    import numpy as np
    return np.random.rand(n)


def fun(data):
    s = sorted(data)


def cleanup():
    pass


if __name__ == '__main__':
    result = complexity.approximate(setup, fun, cleanup, timeout=20)
    print(result)
    print(result['complexity'])
    get_time = result['time_model']
    get_size = result['size_model']
    print(get_time(200))
    print(get_size(1))
