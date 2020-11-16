import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('seaborn')

def smooth_range(arr):
    minVal = arr[0]
    minInd = 0
    for i in range(0, len(arr)):
        if arr[i] < minVal:
            minVal = arr[i]
            minInd = i
        arr[i] = minVal
    
    return arr, minVal, minInd

def plot_training_error(files, begin = 0, end = 0, show_valid_err = True):

    column = 1 if show_valid_err else 0

    for (file, label) in files:
        errors = np.genfromtxt(fname=file,
                                dtype=np.float32,
                                delimiter=',',
                                skip_header=0)[:, column]
        length = np.size(errors, 0)

        if end == 0:
            end = length
        #if end >= length:
        #    np.append(errors, [errors[-1]] * (end - length))

        smoothErrors, minVal, minInd = smooth_range(errors[begin:end])
        plt.plot(smoothErrors, label=label)
        print("min: {}, {} - {}".format(minVal, minInd, label))

    plt.legend()
    plt.show()

plot_training_error([("loss_hamiltonian.txt", "Hamiltonian"),
                 #    ("loss_mlp.txt", "MLP"),
                     ("loss_antisym.txt", "AntiSym")],
                     0, 3000, True)