import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="mse.txt",
                       dtype=np.float32,
                       delimiter=',',
                       skip_header=0)

end = -128
energy = data[:end,0]
errors = data[:end,4:-4]
num = errors.shape[1]

for i in range(0, num):
	plt.semilogy( energy, errors[:,i])
plt.xlabel('initial energy'), plt.ylabel('error')
plt.legend( ["Verlet", "RK2", "RK3", "RK4"])
#plt.legend( ["leap frog", "ResNet", "AntiSym"])
plt.show()