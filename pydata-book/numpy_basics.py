from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Numpy ndarray:  multidimensional array object
data = randn(2, 3)
print(data)
print(data * 10)
print(data + data)
print(data.shape)
print(data.dtype)

# Creating ndarrays
data1 = [6, 7.5, 8, 0, 1]   # Python list
arr1 = np.array(data1) # Numpy 1-d array
print(arr1)

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]] # Python nested list
arr2 = np.array(data2) # Numpy 2-d array
print(arr2)
print(arr2.ndim)
print(arr2.shape)
print(arr1.dtype)
print(arr2.dtype)

# Arrays of zeros, ones, empty, and ranges
zeros1 = np.zeros(10)
zeros2 = np.zeros((3, 6))
empty1 = np.empty((2, 3, 2))
ones1 = np.ones((4, 5))
x1 = np.arange(15)

# Specifying data types for ndarrays
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
print(arr1.dtype)
print(arr2.dtype)

# Implicit type definition based on array contents
arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)

# Casting from one type to another with astype()
float_arr = arr.astype(np.float64)
print(float_arr.dtype)

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
int_arr = arr.astype(np.int32)
print(arr)
print(int_arr)

# Converting numeric strings to numbers
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
arr = numeric_strings.astype(float) # Can leave out the 64

int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
arr = int_array.astype(calibers.dtype)

empty_uint32 = np.empty(8, dtype='u4')
print(empty_uint32)

# Operations between arrays and scalars
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print(arr)
print(arr * arr)
print(arr - arr)
print(1 / arr)
print(arr ** 0.5)

# ----------------------------------------------------------------------
# Basic indexing and slicing
# -- For 1-d arrays
arr = np.arange(10)
print(arr)
print(arr[5])
print(arr[5:8])
arr[5:8] = 12
print(arr)

# -- For 2-d arrays
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[2])
print(arr2d[0][2])
print(arr2d[0, 2])

# -- For 3-d arrays
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d)
print(arr3d[0])
old_values = arr3d[0].copy()
arr3d[0] = 42
print(arr3d)
arr3d[0] = old_values
print(arr3d)
print(arr3d[1, 0])

# Indexing with slices
print(arr)
print(arr[1:6])
print(arr2d)
print(arr2d[:2])
print(arr2d[:2, 1:])
print(arr2d[1, :2])
print(arr2d[2, :1])
print(arr2d[:, :1])
arr2d[:2, 1:] = 0
print(arr2d)

# Boolean indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = randn(7, 4)
print(names)
print(data)
print(names == 'Bob')
print(data[names == 'Bob', 2:])
print(data[names == 'Bob', 3])
print(names != 'Bob')
print(data[-(names == 'Bob')])
mask = (names == 'Bob') | (names == 'Will')
print(mask)
print(data[mask])
data[data < 0] = 0
print(data)
data[names != 'Joe'] = 7
print(data)

# Fancy indexing
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
print(arr)
print(arr[[4, 3, 0, 6]])
print(arr[[-3, -5, -7]])

arr = np.arange(32).reshape((8, 4))
print(arr)
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])
print(arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]])
print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])

# ----------------------------------------------------------------------
# Transposing arrays and swapping axes
arr = np.arange(15).reshape((3, 5))
print(arr)
print(arr.T) # Transpose

arr = np.random.randn(6, 3)
print(arr)
print(np.dot(arr.T, arr))

arr = np.arange(16).reshape((2, 2, 4))
print(arr)
print(arr.transpose((1, 0, 2)))
print(arr.swapaxes(1, 2))

# ----------------------------------------------------------------------
# Universal functions:  fast element-wise array functions
arr = np.arange(10)
print(np.sqrt(arr))
print(np.exp(arr))
x = randn(8)
y = randn(8)
print(x)
print(y)
print(np.maximum(x, y)) # element-wise maximum
arr = randn(7) * 5
print(np.modf(arr))

# ----------------------------------------------------------------------
# Data processing using arrays
points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs ** 2 + ys ** 2)
print(z)
plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
plt.draw()

# ----------------------------------------------------------------------
# Expressing conditional logic as array operations
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result
result = np.where(cond, xarr, yarr)
result
arr = randn(4, 4)
arr
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr) # set only positive values to 2

'''
# Not to be executed
result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)

# Not to be executed
np.where(cond1 & cond2, 0,
         np.where(cond1, 1,
                  np.where(cond2, 2, 3)))

# Not to be executed
result = 1 * cond1 + 2 * cond2 + 3 * -(cond1 | cond2)
'''
# ----------------------------------------------------------------------
# Mathematical and statistical methods
arr = np.random.randn(5, 4) # normally-distributed data
arr.mean()
np.mean(arr)
arr.sum()
arr.mean(axis=1)
arr.sum(0)
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0)
arr.cumprod(1)

# ----------------------------------------------------------------------
# Methods for boolean arrays
arr = randn(100)
(arr > 0).sum() # Number of positive values
bools = np.array([False, False, True, False])
bools.any()
bools.all()

# ----------------------------------------------------------------------
# Sorting
arr = randn(8)
arr
arr.sort()
arr
arr = randn(5, 3)
arr
arr.sort(1)
arr
large_arr = randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))] # 5% quantile

# ----------------------------------------------------------------------
# Unique and other set logic
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)
sorted(set(names))
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

# ----------------------------------------------------------------------
# File input and output with arrays

# Storing arrays on disk in binary format
arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')
np.savez('array_archive.npz', a=arr, b=arr)
arch = np.load('array_archive.npz')
arch['b']
'''
!rm some_array.npy
!rm array_archive.npz
'''

# Saving and loading text files
'''
!cat array_ex.txt
arr = np.loadtxt('array_ex.txt', delimiter=',')
arr
'''

# ----------------------------------------------------------------------
# Linear algebra

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
y
x.dot(y)  # equivalently np.dot(x, y)

np.dot(x, np.ones(3))
np.random.seed(12345)

from numpy.linalg import inv, qr
X = randn(5, 5)
mat = X.T.dot(X)
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
r

# ----------------------------------------------------------------------
# Random number generation

samples = np.random.normal(size=(4, 4))
samples

from random import normalvariate
N = 1000000
'''
%timeit samples = [normalvariate(0, 1) for _ in xrange(N)]
%timeit np.random.normal(size=N)
'''

# ----------------------------------------------------------------------
# Example: Random Walks

import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

np.random.seed(12345)
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
walk.min()
walk.max()
(np.abs(walk) >= 10).argmax()

# Simulating many random walks at once
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks

walks.max()
walks.min()

hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum() # Number that hit 30 or -30
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()
steps = np.random.normal(loc=0, scale=0.25,
                         size=(nwalks, nsteps))
