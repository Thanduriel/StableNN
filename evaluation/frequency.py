import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="frequency.txt",
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)

state = data[:,0]
frequencies = data[:,3:]
num = frequencies.shape[1]

for i in range(0, num):
	plt.plot( state, np.abs(frequencies[:,i]))
plt.xlabel('initial state'), plt.ylabel('period length')
plt.legend( ["τ", "τ / 2", "τ / 4"])
plt.show()