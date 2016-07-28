from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

print('Hello world!')
print(1 + 2)

for i in range(5):
    if i == 3:
        print('Kittens')
    else:
        print(i)


x = np.arange(10)
y = x + 2
plt.figure()
plt.plot(x, y)
plt.show()