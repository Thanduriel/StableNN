import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

def isFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

pointSets = [[],[],[],[],[]]

with open('freqs2.txt') as f:
	nets = []
	for line in f:
		nums = line.split(',')[:-1]
		if len(nums) > 0:
			if not isFloat(nums[0]):
				nums = nums[1:]
			nums = [float(x) for x in nums]
			nets.append(nums)
for freqs in nets:
	for i in range(1, len(freqs)):
		if freqs[i] > 0.0:
			pointSets[len(freqs)-1].append([freqs[0], freqs[i]])

array1 = np.array(pointSets[1]);
array2 = np.array(pointSets[2]);
array3 = np.array(pointSets[3]);

fig1, ax1 = plt.subplots()
shift = 0.05
plt.plot( array1[:,0]*(1.0-shift), 1.0 / array1[:,1], "b+")
plt.plot( array2[:,0], 1.0 / array2[:,1], "r+")
plt.plot( array3[:,0]*(1.0+shift), 1.0 / array3[:,1], "y+")

ticks = np.unique(array1[:,0])#np.array([0.1, 0.05, 0.025, 0.01, 0.005])
if ticks[-1] / ticks[0] > 10:
	ax1.set_xscale('log')
ax1.set_xticks(ticks)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel('time step')
plt.ylabel('frequency')
ax1.legend(["1", "2", "3"])

plt.show()
	
	
