import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

data = np.genfromtxt(fname="../build/mse.txt",
                       dtype=np.float32,
                       delimiter=',',
                       skip_header=0)

end = 4096
energy = data[1:end,0]
errors = data[1:end,3:]
num = errors.shape[1]

diffusion = np.linspace(0.05,3.0, 59)

for i in range(0, num):
	plt.semilogy( diffusion, errors[:,i])
plt.xlabel('diffusion coefficient'), plt.ylabel('error')
plt.legend( ["Finite Difs", "Conv1", "conv2", "conv3", "conv4", "TCN", "TCN Avg", "TCN No Res"])

axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([0.0001,10.0])

plt.show()