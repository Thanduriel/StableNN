import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.style.use('seaborn')

names = ["green.txt", "heateq_adam3.txt", "heateq3.txt"]
#names = ["heateq1.txt", "heateq2.txt", "heateq3.txt"]
markers = ["o", "s", "."]

data = []
for file in names:
	data.append(np.genfromtxt(fname=file,
                       dtype=np.float32,
                       delimiter=' ',
                       skip_header=0)
)

for i in range(0, len(data)):
	kernel = data[i]
	halfSize = len(kernel) // 2
	plt.semilogy( np.linspace(-halfSize, halfSize, len(kernel)), kernel + 0.005, marker=markers[i], linestyle='-', linewidth=1)
plt.xlabel('index'), plt.ylabel('coefficient + 0.005')
plt.legend( names )
plt.show()