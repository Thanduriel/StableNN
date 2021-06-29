import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/dft_coefs_rand.txt",
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)

num = data.shape[1]
coef = 8
n = 17
integrator = 0

for i in [0,3,10,13,16]: # 1,4,7,10,13,16
	plt.semilogy( data[:,i+1])
	plt.semilogy( data[:,i+1+3*17], linestyle='dashed')
#for i in range(n*integrator, n*(integrator+1)):
#    plt.semilogy( data[:,0],data[:,i+1])
plt.xlabel('time step'), plt.ylabel('norm')
#plt.legend( ["Finite Difs", "CNN", "CNNnoB", "CNNsym", "TCN", "TCN Avg", "TCN No Res"])
plt.show() # "Analytic", "FD", 