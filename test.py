import numpy as np

kernel = np.zeros((3,3))

sigma = 1.3

for yp in range(-1,2,1):
    for xp in range(-1,2,1):
        kernel[xp+1,yp+1] = np.exp(-(np.power(xp,2) + np.power(yp,2))/(2*np.power(sigma,2)))/((np.sqrt(2*np.pi))*sigma)

print(kernel)

v = 0
for i in range(3):
    for j in range(3):
        v += kernel[i,j]

print(v)

kernel2 = kernel/v

print(kernel2)