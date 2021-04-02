# NumPy

NumPy is a library for multidimensional array objects.

import numpy as np

x = np.array([1, 2, 3])
x

2 * x

np.shape(x)

y = np.array([[4], [5], [6]])
y

x + y

z = x * y

z

np.shape(z)

np.savez_compressed('nums.npz', x=x, y=y)

nums = np.load('nums.npz')

[key for key in nums.keys()]

nums['x']

nums['x'] == x

np.linspace(0, 10, 11)

np.arange(10)

np.arange(0, 10, 0.5)

x = np.arange(10)
y = np.arange(10)
xx, yy = np.meshgrid(x, y)
print(np.shape(xx))
xx

np.add(x, y)

np.multiply(x, y)

np.mean(x)

np.nan

np.nanmean([0, 1, 2, 3, np.nan])

np.reshape(x, (2, 5))

np.where(x < 5, x, x * 10)

For more information, see the [documentation](https://numpy.org/).

