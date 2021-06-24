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

#for i in range(0, num):
#	plt.plot( states[:,i])
plt.plot( states[:,0])
plt.plot( states[:,2])
plt.plot( states[:,3])
#plt.plot( states[:,6])
#plt.plot( states[:,7])
#plt.plot( states[:,8])
plt.xlabel('time step'), plt.ylabel('error')
plt.legend( ["Finite Difs", "CNN", "CNNnoB", "CNNsym", "TCN", "TCN Avg", "TCN No Res"])
plt.show() # "Analytic", "FD", 