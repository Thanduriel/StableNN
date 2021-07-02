import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

files = []
names = ["../build/energy_s_01_long.txt", "../build/energy_-1_1.txt", "../build/energy_rand_1m.txt", "../build/energy_sym_long.txt", "../build/energy.txt"]
#for i in range(2,4):
#	names.append("../build/energy_{}.txt".format(i))
#names = ["../build/energy_0_1.txt", "../build/energy_0_5.txt", "../build/energy_1_0.txt", "../build/energy_2_0.txt", "../build/energy_2_5.txt", "../build/energy_3_0.txt"]
for name in names:
	files.append(np.genfromtxt(fname=name,
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0))

for data in files:
	steps = data[:,0]
	errors = data[:,3:]
	num = errors.shape[1]

	plt.semilogy( steps, data[:,1], linestyle='dashed')
	plt.semilogy( steps, data[:,4])
#	for i in range(0, num):
#		plt.semilogy( steps, errors[:,i])

plt.xlabel('time step'), plt.ylabel('error')
plt.legend( ["Reference", "Finite Difs", "CNN", "CNNnoB", "CNNsym", "TCN", "TCN Avg", "TCN No Res"])
plt.show() # "Analytic", "FD", 