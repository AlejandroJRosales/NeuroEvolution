import numpy as np
import neuro_evolution as ne


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

X = np.array([[1, 1, 0],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [0, 1, 0]
              ])

Y = np.array([[1, 0, 0, 1, 0, 1, 0, 1]]).T

weights = ne.train(X, Y, test_data=[0, 1, 1], correct=0)
print("weights: ", end="")
for weight in weights:
    print(weight, end="")
print()
print(nonlin(np.dot(X, weights)))
