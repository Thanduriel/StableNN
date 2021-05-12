import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/mse.txt",
                       dtype=np.float32,
                       delimiter=',',
                       skip_header=0)

end = -2
energy = data[1:end,0]
errors = data[1:end,8:]
num = errors.shape[1]

for i in range(0, num):
	plt.semilogy( energy, errors[:,i])
plt.xlabel('initial energy'), plt.ylabel('error')
#plt.legend( ["Verlet", "RK2", "RK3", "RK4"])
plt.legend( ["0", "01", "001"])

axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0.0001,10.0])

plt.show()