import numpy as np

a1 = np.arange(15).reshape(3, 5)
print(a1.shape, a1.ndim, a1.dtype)

print(np.zeros((2, 3)))
print(np.ones((2, 3, 4)))

print(np.linspace(0, 2*np.pi, 100))
