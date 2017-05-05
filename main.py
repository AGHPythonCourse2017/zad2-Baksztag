from complexity import model


def setup(n):
    import numpy as np
    return np.random.rand(n, n)


def fun(data):
    for i in data:
        for j in i:
            j += 1


def cleanup():
    pass

result = model.get_model(setup, fun, cleanup)
print(result)
