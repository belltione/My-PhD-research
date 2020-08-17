from file_loading import get_File, bytes_feature, int64_feature, convert_to_TFRecord
import tensorflow as tf
import numpy as np


def main():

    #if tf.test.gpu_device_name():
    #    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    #else:
    #    print("Please install GPU version of TF")
    
    SUBJECT_NUM = 8
    for i in np.arange(1, SUBJECT_NUM + 1):
        # IMPORTANT:Put all training images in the same dictionary
        dataset_dir = '/home/eric/workload_project/workload_image_data/'
        train_tfrecord_dir = '/home/eric/workload_project/Train' + str(i) +'.tfrecords'
        val_tfrecord_dir = '/home/eric/workload_project/Val' + str(i) +'.tfrecords'
        #val_sub_num = i
  
        train_images, train_labels, val_images, val_labels  = get_File(dataset_dir, i)
        #print(val_labels)
        #print(val_images)
        convert_to_TFRecord(train_images, train_labels, train_tfrecord_dir, i)
        print('Number of training images: %d' % (len(train_images)))
        convert_to_TFRecord(val_images, val_labels, val_tfrecord_dir, i)
        print('Number of validation images: %d' % (len(val_images)))

    gpu_device_name = tf.test.gpu_device_name()
    print(gpu_device_name)
if __name__ == '__main__':
    main()
