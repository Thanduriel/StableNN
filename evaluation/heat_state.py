import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/state.txt",
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)

diffusion = data[:,0]
states = data[:,1:]
num = states.shape[1]

for i in range(7, num, 4):
	plt.plot( states[:,i])
#plt.plot( states[:,8])
#plt.plot( states[:,11])
#plt.plot( states[:,3])
plt.xlabel('time step'), plt.ylabel('u')
plt.legend( ["Finite Difs", "CNN", "CNNnoB", "CNNsym", "TCN", "TCN Avg", "TCN No Res"])
plt.show() # "Analytic", "FD", 