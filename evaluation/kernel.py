import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

names = ["linear/green.txt", "linear/0_kernel.txt", "linear/1_kernel.txt", "linear/2_kernel.txt","linear/3_kernel.txt", "linear/4_kernel.txt", "linear/5_kernel.txt", "linear/6_kernel.txt",
"linear/7_kernel.txt", "linear/8_kernel.txt", "linear/9_kernel.txt","linear/10_kernel.txt", "linear/11_kernel.txt", "linear/12_kernel.txt", "linear/13_kernel.txt", "linear/14_kernel.txt"]
#names = ["heateq1.txt", "heateq2.txt", "heateq3.txt"]
markers = ["o", "s", "."]

N = 32

data = [np.array(np.linspace(-N//2+1, N//2-1, N-1))]
for file in names:
	kernel = np.genfromtxt(fname=file,
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)
	halfSize = len(kernel) // 2
	padding = N // 2 - halfSize - 1
	kernel = np.insert(kernel, 0, padding * [float('inf')])
	kernel = np.insert(kernel, len(kernel), padding * [float('inf')])
	data.append(kernel)
	
data = np.array(data)
np.savetxt("kernels.txt", np.transpose(data)) # np.concatenate((index,data),axis=0)
exit()

def processKernel(kernel):
	signs = np.sign(kernel)
	kernel = np.abs(kernel)
	min = np.min(kernel)
#	kernel /= min
	kernel = np.log10(kernel + 1)
	return kernel * signs

for i in range(0, len(data)):
	kernel = data[i]
	halfSize = len(kernel) // 2
	plt.plot( np.linspace(-halfSize, halfSize, len(kernel)), processKernel(kernel), marker=markers[i], linestyle='-', linewidth=1)
plt.xlabel('index'), plt.ylabel('coefficient + 0.005')
plt.legend( names )
plt.show()