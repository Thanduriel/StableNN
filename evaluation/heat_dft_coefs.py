import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/dft_coefs.txt",
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)

num = data.shape[1]
coef = 8
n = 17
integrator = 3

#plt.semilogy( data[:,coef])
#plt.semilogy( data[:,coef+17])
for i in range(n*integrator, n*(integrator+1)):
    plt.semilogy( data[:,i])
plt.xlabel('time step'), plt.ylabel('norm')
#plt.legend( ["Finite Difs", "CNN", "CNNnoB", "CNNsym", "TCN", "TCN Avg", "TCN No Res"])
plt.show() # "Analytic", "FD", 