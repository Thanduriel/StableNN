import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="large_nets_avg_error.txt",
					   dtype=np.float32,
					   delimiter=' ',
					   skip_header=0)

steps = data[:,0]
errors = data[:,2:]
num = errors.shape[1]

for i in range(0, num):
	if i != 1:
		plt.plot( steps,errors[:10000,i])
plt.xlabel('time step'), plt.ylabel('error')
plt.legend( ["Verlet", "RK4", "ResNet", "ResNet26", "ResNet44"])
plt.show()