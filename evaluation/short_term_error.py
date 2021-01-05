import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="mse.txt",
                       dtype=np.float32,
                       delimiter=',',
                       skip_header=0)

energy = data[:,0]
errors = data[:,2:-1]
num = errors.shape[1]

for i in range(0, num):
	plt.semilogy( energy, errors[:,i])
plt.xlabel('initial energy'), plt.ylabel('error')
plt.legend( ["verlet", "hamiltonian"])
plt.show()