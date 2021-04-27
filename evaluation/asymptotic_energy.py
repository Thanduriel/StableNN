import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/asymptotic.txt",
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)

timeStep = data[:,0]
borders = data[:,1:]
num = borders.shape[1]

for i in range(0, num):
	plt.plot( timeStep, borders)
plt.xlabel('time step'), plt.ylabel('energy')
plt.show()