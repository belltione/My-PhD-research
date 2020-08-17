import tensorflow as tf
from sklearn.metrics import confusion_matrix


def read_and_decode(filename, BATCH_SIZE, MAX_EPOCH): 
    
    filename_queue = tf.train.string_input_producer([filename], 
                                                    num_epochs=MAX_EPOCH)
    
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    
    img_features = tf.parse_single_example(
            serialized_example,
            features={ 'Label'    : tf.FixedLenFeature([], tf.int64),
                       'image_raw': tf.FixedLenFeature([], tf.string), })
    
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    #print 'type of image ='
    
    
    image = tf.reshape(image, [32, 32, 3])
    
    label = tf.cast(img_features['Label'], tf.int64)
    #print type(image)
    #print 'size of image ='
    #print image.get_shape()
    #print 'size of label ='
    #print label.get_shape()    
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch =tf.train.shuffle_batch(
                                 [image, label],
                                 batch_size = BATCH_SIZE,
                                 capacity = 1000 + 3 * BATCH_SIZE,
                                 min_after_dequeue = 1000)

    

    return image_batch, label_batch

def Weight(shape, mean=0, stddev=1):
    init = tf.truncated_normal(shape, mean=mean, stddev=stddev)
    return tf.Variable(init)

def bias(shape, mean=0, stddev=1):
    init = tf.truncated_normal(shape, mean=mean, stddev=stddev)
    return tf.Variable(init)

def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')


def print_confusion_matrix(input_matrix, class_num):
    COLUMN_LENGTH = 8 # Define the length of column for display
    output_str = ''

    print ("Confusion matrix: ")
    output_str = output_str + output_str.rjust(COLUMN_LENGTH - 1,' ') + '|'    
    for k in range(0, class_num):
        temp = 'class' + str(k + 1) + '|'
        output_str =  output_str +  temp.rjust(COLUMN_LENGTH,' ')
    print output_str
    for dash in range(0, class_num):  #print the dash line
        output_str = '' 
        output_str =  output_str +  output_str.rjust(COLUMN_LENGTH * (class_num + 1),'-')
    print output_str   

    for i in range(0, class_num):
        output_str = ''
        temp = 'class' + str(i + 1) + '|'
        output_str =  output_str +  temp.rjust(COLUMN_LENGTH,' ')       
        for j in range(0, class_num):
            temp = input_matrix[i][j].astype(str) + '|'
            output_str =  output_str +  temp.rjust(COLUMN_LENGTH,' ')
        print output_str 
        for dash in range(0, class_num): #print the dash line
            output_str = '' 
            output_str =  output_str +  output_str.rjust(COLUMN_LENGTH * (class_num + 1),'-')
        print output_str 


# ========================================================================#
#                              Main program                               #
# ========================================================================#
    
#filename = './Train.tfrecords'
train_filename = './Train1.tfrecords'
val_filename = './Val1.tfrecords'
output_model_path = '/home/eric/workload_project'
BATCH_SIZE = 64
MAX_EPOCH = 40
LABEL_NUM = 3
TOTAL_SUB_NUM = 2484
VAL_SUB_NUM = 359
VAL_ITERATION = 50

# Input images & labels
X  = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape = [None, LABEL_NUM])


# Construct the CNN model
# Conv1
W_conv1 = Weight([3, 3, 3, 16])
b_conv1 = bias([16])
y_conv1 = conv2d(X, W_conv1) + b_conv1

# ReLU1
relu1 = tf.nn.relu(y_conv1)

# Pool1
pool1 = max_pool(relu1)

# Conv2
W_conv2 = Weight([3, 3, 16, 32])
b_conv2 = bias([32])
y_conv2 = conv2d(pool1, W_conv2) + b_conv2

# ReLU2
relu2 = tf.nn.relu(y_conv2)

# Pool2
pool2 = max_pool(relu2)

# FC1
W_fc1 = Weight([8*8*32, 500])
b_fc1 = bias([500])
h_flat = tf.reshape(pool2, [-1, 8*8*32])
y_fc1 = tf.matmul(h_flat, W_fc1) + b_fc1

# ReLU3
relu3 = tf.nn.relu(y_fc1)

# FC2 - Output layer
W_fc2 = Weight([500, 3]) # Here we have 3 classes
b_fc2 = bias([3])

y = tf.matmul(relu3, W_fc2) + b_fc2

# Cost optimizer
train_lossFcn = tf.nn.softmax_cross_entropy_with_logits
val_lossFcn = tf.nn.softmax_cross_entropy_with_logits
#lossFcn = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
train_cost = tf.reduce_mean(train_lossFcn(labels=y_, logits=y))
val_cost = tf.reduce_mean(val_lossFcn(labels=y_, logits=y))

# Use AdamOptimizer for the optimization task
train_step = tf.train.AdamOptimizer(0.0001).minimize(train_cost)

#calculate the accuracy
train_correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
val_correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
val_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))
cf_matrix = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(y_, 1))
# op2 = tf.confusion_matrix(y_true, y_pred, num_classes=2, dtype=tf.float32, weights=tf.constant([0.3, 0.4, 0.3]))


# Loading the input data
train_img_bat, train_lb_bat = read_and_decode(train_filename, BATCH_SIZE, MAX_EPOCH)
val_img_bat, val_lb_bat = read_and_decode(val_filename, VAL_SUB_NUM, MAX_EPOCH * (TOTAL_SUB_NUM / VAL_SUB_NUM))

#lb_bat = tf.Print(lb_bat, [lb_bat], message="This is lb_bat: ")

train_x = tf.reshape(train_img_bat, [-1, 32, 32, 3])
train_y = tf.one_hot(train_lb_bat, LABEL_NUM)
val_x = tf.reshape(val_img_bat, [-1, 32, 32, 3])
val_y = tf.one_hot(val_lb_bat, LABEL_NUM)
saver = tf.train.Saver() 


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    i = 0
    try:
        while not coord.should_stop():
            image_train, label_train = sess.run([train_x, train_y])
            image_val, label_val = sess.run([val_x, val_y])
            sess.run(train_step, feed_dict={ X : image_train, 
                                             y_: label_train})
            
            if i % VAL_ITERATION == 0:
                #train_prediction_score = sess.run(train_accuracy, feed_dict={ X  : image_train, 
                #                                                y_ : label_train})
                #val_prediction_score = sess.run(train_accuracy, feed_dict={ X  : image_val, 
                #                                                y_ : label_val})    

                # calculate the confusion matrix           
                cf_mat = sess.run(cf_matrix, feed_dict={ X  : image_val, 
                                                                y_ : label_val})
                # calculate the accuracy from the confusion matrix
                cf_mat_trace = tf.trace(tf.cast(cf_mat,tf.float32))
                cf_mat_sum = tf.reduce_sum(tf.cast(cf_mat,tf.float32))
                accuracy =  tf.divide(cf_mat_trace, cf_mat_sum)
                pred_score = sess.run(accuracy)

                print('Iter %d, accuracy %4.2f%%' % (i,pred_score*100))
                # print the confusion matrix
                print_confusion_matrix(cf_mat, LABEL_NUM)

                #print('Iter %d, accuracy %4.2f%%' % (i,val_prediction_score*100))
                
                
            i += 1
            
    except tf.errors.OutOfRangeError:
        print('Done!')

        # Save the training model
        saver.save(sess, output_model_path +'/workload_model', global_step=i)
        
    finally:
        coord.request_stop()
            
    coord.join(threads)


