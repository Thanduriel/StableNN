import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

positions = np.genfromtxt(fname="spacetime.csv",
                       dtype=np.float32,
                       delimiter=',',
                       skip_header=0)

numIntegrators = positions.shape[1]
print(numIntegrators)

for i in range(0, numIntegrators):
	plt.plot( positions[:100,i])
plt.xlabel('t'), plt.ylabel('x')
plt.legend( ["reference", "hamiltonian", "antisymmetric", "hamiltonian2"], loc=2)
plt.show()