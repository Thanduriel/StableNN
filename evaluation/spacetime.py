import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

positions = np.genfromtxt(fname="spacetime.txt",
                       dtype=np.float32,
                       delimiter=',',
                       skip_header=0)

numIntegrators = positions.shape[1]-1

for i in range(0, numIntegrators):
	plt.plot( positions[:300,i])
plt.xlabel('t'), plt.ylabel('x')
plt.legend( ["leap frog", "forward euler", "network"], loc=2)
plt.show()