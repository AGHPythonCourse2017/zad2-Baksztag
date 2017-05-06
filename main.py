from complexity import complexity


def setup(n):
    import numpy as np
    return np.random.rand(n)


def fun(data):
    l = data[:]


def cleanup():
    pass

if __name__ == '__main__':
    result = complexity.approximate(setup, fun, cleanup)
    print(result)
