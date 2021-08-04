import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/random_avg_error.txt", # globalerror / relative_error / random_avg_error
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)

steps = data[:,0]
errors = data[:,3:]
num = errors.shape[1]

for i in range(0, num):
	plt.semilogy( steps, errors[:10000,i])
plt.xlabel('time step'), plt.ylabel('error')
plt.legend( ["Finite Difs", "CNN", "CNNnoB", "CNNsym", "TCN", "TCN Avg", "TCN No Res"])
plt.show() # "Analytic", "FD", 