from complexity import complexity


def setup(n):
    import numpy as np
    return np.random.rand(n)


def fun(data):
    list(data)


def cleanup():
    pass


if __name__ == '__main__':
    result = complexity.approximate(setup, fun, cleanup, time=20)
    print(result)
