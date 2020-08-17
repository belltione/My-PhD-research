import os
import cv2
import numpy as np
import tensorflow as tf



def get_File(file_dir, val_sub_num):
    # The images in each subfolder
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    # The subfolders
    train_subfolders = []
    val_subfolders = []

    # Using "os.walk" function to grab all the files in each folder
    for dirPath, dirNames, fileNames in os.walk(file_dir):        
        for name in fileNames: 
            #print name
            train_val_index = train_val_data_split(name, val_sub_num)  
            if train_val_index == 0:
                train_images.append(os.path.join(dirPath, name))
                train_labels.append(name[0])  # Get the label of image data from its name
            else:
                val_images.append(os.path.join(dirPath, name))
                val_labels.append(name[0])  # Get the label of image data from its name
    
            for name in dirNames:
                if train_val_index == 0:
                    train_subfolders.append(os.path.join(dirPath, name))
                else:
                    val_subfolders.append(os.path.join(dirPath, name))

    # To record the labels of the image dataset
    
#    count = 0
#    for a_folder in train_subfolders:
#        n_img = len(os.listdir(a_folder))
#        #labels = np.append(labels, n_img * [count])
#        count+=1
    
    train_subfolders = np.array([train_images, train_labels])   
    train_subfolders = train_subfolders.transpose()

    train_image_list = list(train_subfolders[:, 0])
    train_label_list = list(train_subfolders[:, 1])
    train_label_list = [int(float(i)) for i in train_label_list]

    val_subfolders = np.array([val_images, val_labels])   
    val_subfolders = val_subfolders.transpose()

    val_image_list = list(val_subfolders[:, 0])
    val_label_list = list(val_subfolders[:, 1])
    val_label_list = [int(float(i)) for i in val_label_list]
    
    return train_image_list, train_label_list, val_image_list, val_label_list
   


def train_val_data_split(image_name, val_sub_num):
    sub_num1 = image_name[2]
    sub_num2 = image_name[2] + image_name[3]
    
    #print type(sub_num2.isdigit())
    
    if sub_num2.isdigit() == True:
        
        if sub_num2 == str(val_sub_num):
            
            return 1
        else:
            return 0
    elif sub_num1.isdigit() == True:
        if sub_num1 == str(val_sub_num):           
            return 1
        else:
            return 0
    else:
        print "Failed to get the subject number of input data."




def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_TFRecord(images, labels, filename, subject_num):
    n_samples = len(labels)
    TFWriter = tf.python_io.TFRecordWriter(filename)
    
    print('\nSubject %d transform start...' % (subject_num))
    for i in np.arange(0, n_samples):
        try:
            image = cv2.imread(images[i], cv2.IMREAD_UNCHANGED)

            if image is None:
                print('Error image:' + images[i])
            else:
                image_raw = image.tostring()

            label = int(labels[i])
            
            
            ftrs = tf.train.Features(
                    feature={'Label': int64_feature(label),
                             'image_raw': bytes_feature(image_raw)}
                   )
        
            
            example = tf.train.Example(features=ftrs)

            
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')

 
    TFWriter.close()
    print('Subject %d transform done!'% (subject_num))

