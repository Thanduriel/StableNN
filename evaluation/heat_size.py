import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

def getIndex(kernel,channels,depth):
	return kernel + 3 * channels + 9 * depth
	
def getSliceIndex(kernel,channels,depth, variable):
	arr = []
	for i in range(0,3):
		if variable == 0:
			arr.append(getIndex(i,channels,depth))
		if variable == 1:
			arr.append(getIndex(kernel,i,depth))
		if variable == 2:
			arr.append(getIndex(kernel,channels,i))
	return np.array(arr)
	
def getSliceIndexAll():
	arr = []
	for i in range(0,3):
		arr.append(getIndex(i,i,i))
	return np.array(arr)

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/cnn_weights.txt",
					dtype=np.float32,
					delimiter=' ',
					skip_header=0)

loss = data[:,0]
weights = data[:,1]

'''
indices = getSliceIndex(1,1,1,0)
plt.semilogy( weights[indices], loss[indices])

indices = getSliceIndex(1,1,1,1)
plt.semilogy( weights[indices], loss[indices])

indices = getSliceIndex(1,1,1,2)
plt.semilogy( weights[indices], loss[indices])

indices = getSliceIndexAll()
plt.semilogy( weights[indices], loss[indices])
'''

arr = []
for j in range(0,3):
	for i in range(0,3):
		indices = getSliceIndex(i,i,i,j)
		plt.semilogy( weights[indices], loss[indices])
		arr.append(weights[indices])
		arr.append(loss[indices])
np.savetxt("cnn_size_processed.txt", np.transpose(np.array(arr)))

plt.xlabel('weights'), plt.ylabel('loss')
plt.legend( ["kernel", "channels", "depth", "all"])
plt.show() # "Analytic", "FD", 