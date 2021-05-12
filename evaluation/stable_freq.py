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

withLines = True

pointSets = [[],[],[],[],[]]
names = []
learningRates = [[],[],[],[],[]]

# parse file
with open('freqsSingle.txt') as f:
	nets = []
	for line in f:
		nums = line.split(',')[:-1]
		if len(nums) > 0:
			# first row is network name
			if not isFloat(nums[0]):
				#first part of the name is the learning rate
				names.append(int(nums[0].split('_')[0]))
				nums = nums[1:]
			nums = [float(x) for x in nums]
			nets.append(nums)
for j in range(0,len(nets)):
	freqs = nets[j]
	# group by number of attractors
	for i in range(1, len(freqs)):
		if freqs[i] > 0.0:
			idx = len(freqs)-1 # num attractors
			# freqs[0] is the time step
			pointSets[idx].append([freqs[0], freqs[i]])
			if withLines:
				learningRates[idx].append(names[j])

if withLines:
	learningRates = [[], np.array(learningRates[1]),np.array(learningRates[2])]
array1 = np.array(pointSets[1]);
array2 = np.array(pointSets[2]);
array3 = np.array(pointSets[3]);

fig1, ax1 = plt.subplots()
shift = 0.05

if withLines:
	maxLr = np.max(learningRates[1])
	for i in range(0, maxLr):
		arr = np.sort(array1[learningRates[1]==i], axis=0)
		plt.plot( arr[:,0], 1.0 / arr[:,1], color="b", marker = 'o')
		if len(array2) > 0:
			arr = np.sort(array2[learningRates[2]==i], axis=0)
			plt.plot( arr[0::2,0], 1.0 / arr[0::2,1], color="r", marker = 'o')
			plt.plot( arr[1::2,0], 1.0 / arr[1::2,1], color="r", marker = 'o')
		
		ax1.legend(["1 attractor", "2 attractor"])
else:
	plt.plot( array1[:,0]*(1.0-shift), 1.0 / array1[:,1], "b+")
	if array2.size > 0:
		plt.plot( array2[:,0], 1.0 / array2[:,1], "r+")
	if array3.size > 0:
		plt.plot( array3[:,0]*(1.0+shift), 1.0 / array3[:,1], "y+")
	ax1.legend(["1", "2", "3"])

ticks = np.unique(array1[:,0])#np.unique(np.concatenate((array1[:,0], array2[:,0])))
#if ticks[-1] / ticks[0] > 10:
#	ax1.set_xscale('log')
ax1.set_xticks(ticks)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel('time step')
plt.ylabel('frequency')

plt.show()
	
	
