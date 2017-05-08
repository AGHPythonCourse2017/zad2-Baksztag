from complexity import complexity


def setup(n):
    import numpy as np
    return np.random.rand(n, n)


def fun(data):
    for i in data:
        for j in i:
            j += 1


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
