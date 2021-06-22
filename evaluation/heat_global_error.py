import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/globalerror.txt", # globalerror / relative_error
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)

steps = data[:,0]
errors = data[:,2:]
num = errors.shape[1]

for i in range(0, num):
	plt.semilogy( errors[:10000,i])
plt.xlabel('time step'), plt.ylabel('error')
plt.legend( ["Finite Difs", "CNN", "CNNnoB", "CNNsym", "TCN", "TCN Avg", "TCN No Res"])
plt.show() # "Analytic", "FD", 