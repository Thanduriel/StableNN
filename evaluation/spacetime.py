import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

state = np.genfromtxt(fname="../build/spacetime.txt",
                       dtype=np.float32,
                       delimiter=',',
                       skip_header=0)

# division also truncates the empty column
numIntegrators = state.shape[1] // 2

labels = ["reference", "leapfrog", "net", "cos"]

maxTime = 128

plt.figure(1)
for i in range(0, numIntegrators):
	plt.plot( state[:maxTime,i*2])
plt.xlabel('t')
plt.ylabel('x')
plt.legend( labels, loc=2)
plt.show()

plt.figure(2)
for i in range(0, numIntegrators):
    plt.plot( state[:maxTime,i*2], state[:maxTime,i*2+1])
plt.xlabel('x')
plt.ylabel('v')
plt.title( 'phase portrait')
plt.legend( labels, loc=2)
plt.show()