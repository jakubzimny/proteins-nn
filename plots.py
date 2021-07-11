import matplotlib.pyplot as plt

batch_size = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
training_time = [8925.37, 4477.1, 2222.7, 1134.1, 546.1, 299.96, 162.77, 91.16, 55.82, 40.72]
training_time_B = [7499.69, 3700.07, 1879.82, 997.04, 499.83, 268.82, 146.75, 86.44, 55.89, 39.65]
gpu_memory = [291, 291, 293, 293, 297, 305, 305, 321, 353, 417]
dataset_fractions = [5, 10, 20, 35, 50, 75, 85, 90]
size_mse = [1.6671, 1.3261, 1.1454, 0.8878, 0.1961, 0.1017, 0.0727, 0.0665]
size_mae = [0.5523, 0.4899, 0.4372, 0.2905, 0.2164, 0.1882, 0.1571, 0.1439]
size_mse_B = [2.1505, 1.8575, 0.4972, 0.3071, 0.2609, 0.2458, 0.2357, 0.2023]
size_mae_B = [0.8307, 0.7815, 0.4968, 0.4068, 0.3932, 0.3851, 0.3711, 0.3465]

seq_lens = [65, 87, 110, 193, 463, 538, 718, 1555, 2166, 4177, 6285]
data_time_A = [ 0.00093, 0.00108, 0.00133,  0.00209, 0.00482, 0.00544, 0.00889, 0.05866, 0.06781, 0.10058, 0.10867]
data_time_B = [0.00099, 0.00095, 0.00107, 0.00185, 0.00418, 0.00522, 0.00641, 0.06759, 0.06888, 0.07731, 0.09939]
model_time_A = [0.71895, 0.72216, 0.73764, 0.72527, 0.72911, 0.75338, 0.76904, 0.70595, 0.73844, 0.73075, 0.71628]
model_time_B = [0.7876, 0.76446, 0.75999, 0.76752, 0.73843, 0.73294, 0.72277, 0.70627, 0.74689, 0.68396, 0.68816]
inference_time_A = [0.37691, 0.39532, 0.41042, 0.40254, 0.43459, 0.40073, 0.42067, 0.42441, 0.45368, 0.54293, 0.58994]
inference_time_B = [0.38154, 0.3753, 0.37714, 0.38927, 0.39473, 0.39819, 0.40172, 0.42542, 0.44459, 0.54016, 0.56027]
total_A = [1.09679, 1.11856, 1.14939, 1.1299, 1.16852, 1.15955, 1.1986, 1.18902, 1.25993, 1.37426, 1.41489]
total_B = [1.17013, 1.14073, 1.13822, 1.15863, 1.13734, 1.13636, 1.13091, 1.19928, 1.26036, 1.30144, 1.34783]

## BS vs time 
# plt.grid()
# plt.title('Influence of batch size on training time (A model)')
# plt.xlabel('Batch size')
# plt.ylabel('Training time [s]') 
# plt.semilogy(batch_size, training_time, marker='x')

## BS vs GPU memory
# plt.grid()
# plt.title('Influence of batch size on GPU memory usage')
# plt.xlabel('Batch size')
# plt.ylabel('GPU Memory used [MB]') 
# plt.semilogx(batch_size, gpu_memory, marker='x')

## Dataset fraction vs test MSE
# plt.grid()
# plt.title('Influence of training set size on test set metrics (Input B)')
# plt.xlabel('Percentage of dataset used for training [%]')
# plt.ylabel('Test set metrics')
# plt.plot(dataset_fractions, size_mse_B, label='MSE', marker='x')
# plt.plot(dataset_fractions, size_mae_B, label='MAE', marker='x')
# plt.legend()

## Scalability
plt.grid()
plt.title('Inference performance scalability (A Input)')
plt.xlabel('Sequence length')
plt.ylabel('Time [s]')
plt.semilogx(seq_lens, data_time_A, label='Data loading time', marker='x')
plt.semilogx(seq_lens, model_time_A, label='Model loading time', marker='x')
plt.semilogx(seq_lens, inference_time_A, label='Inference time', marker='x')
plt.semilogx(seq_lens, total_A, label='Total time', marker='x')
plt.legend()

plt.show()