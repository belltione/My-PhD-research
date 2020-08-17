import scipy.io
import numpy as np
import cv2

input_path = '/home/eric/workload_project/raw_data/'
output_path = '/home/eric/workload_project/workload_image_data/'

raw_data = scipy.io.loadmat(input_path + 's01_051020m_epoch.mat')
input_data = raw_data['data'][:,:,3]


#input_data = 1 / (1 + np.exp(-0.5 * input_data))
#print input_data
#input_data = input_data - np.mean(input_data)
#input_data = input_data / (np.max(input_data) - np.mean(input_data))


cv2.imwrite(output_path + 'test.bmp', input_data)
