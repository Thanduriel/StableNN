import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('seaborn')

def smooth_range(arr):
    smoothArr = np.copy(arr)
    minVal = arr[0]
    minInd = 0
    for i in range(0, len(arr)):
        if arr[i] < minVal:
            minVal = arr[i]
            minInd = i
        smoothArr[i] = minVal
    
    return smoothArr, minVal, minInd
   
prefix = "../build/"
suffix = "_loss.txt"
maxLoss = 5.0
subSample = 2
correctionFactor = 0.01
maxLoss *= correctionFactor
  
def avg_range(arr, window):
    smoothArr = np.copy(arr)
    minVal = arr[0]
    avg = 0
    minInd = 0
    for i in range(0, len(arr)):
        if arr[i] < minVal:
            minVal = arr[i]
            minInd = i
        avg += arr[i]
        if i >= window:
            avg -= arr[i-window]
        smoothArr[i] = min(maxLoss, avg / min(i+1, window))
    
    return smoothArr, minVal, minInd

def plot_training_error(files, begin = 0, end = 0, show_valid_err = True, smooth = 1):

    column = 1 if show_valid_err else 0
    
    data = [np.array(range(0, 1024, subSample))]

    for (file, label) in files:
        errors = np.genfromtxt(fname=prefix + file + suffix,
                                dtype=np.float32,
                                delimiter=',',
                                skip_header=0)[:, column]
        errors *= correctionFactor
        length = np.size(errors, 0)

        if end == 0:
            end = length
        #if end >= length:
        #    np.append(errors, [errors[-1]] * (end - length))

        smoothErrors, minVal, minInd = avg_range(errors[begin:end], smooth)
     #   smoothErrors = np.delete(smoothErrors, list(range(0, smoothErrors.shape[0], 2)), axis=0)
    #    smoothErrors = np.delete(smoothErrors, list(range(0, smoothErrors.shape[0], 2)), axis=0)
        smoothErrors = np.delete(smoothErrors, list(range(0, smoothErrors.shape[0], 2)), axis=0)
        data.append(smoothErrors)
        plt.semilogy(smoothErrors if smooth > 0 else errors[begin:end], label=label)
        print("min: {}, {} - {}".format(minVal, minInd, label))

    data = np.array(data)
 #   data = np.delete(data, list(range(0, data.shape[0], 2)), axis=0)
 #   data = np.delete(data, list(range(0, data.shape[0], 2)), axis=0)
 #   np.savetxt("validation_loss.txt", np.transpose(data), fmt='%1.4e')

    plt.legend()
    plt.show()

plot_training_error([#("tcn/0_5_tcn", "TCN "),
                     #("tcn/0_tcn_no_res", "TCN Avg No Res"),
                     #("tcn/1_4_tcn", "TCN Avg")
					 ("cnn_scale_size/2_2_2_2_cnn_size", "CNN large"), 
					 ("cnn_scale_size/1_1_1_3_cnn_size", "CNN reg"), 
					 ("cnn_scale_size/0_0_0_2_cnn_size", "CNN small"), 
            #         ("conv_repeat/1_1_conv_repeat", "CNN"), 
            #         ("conv_repeat/0_3_conv_repeat", "CNN NoBias"),
            #          ("conv_repeat/7_conv_repeat_sym", "CNN Sym")
            #         ("symmetric/3_conv_sym_res_seed", "CNN Sym"),
            #         ("symmetric/5_conv_res_seed", "CNN"), 
            #         ("con_bias_reg_new/1_2_0_conv_bias_reg", "CNN Reg"),
             #        ("con_bias_reg_new/0_0_1_conv_bias_reg", "CNN NoBias"),
                     ],
                     0, 0, True, subSample)