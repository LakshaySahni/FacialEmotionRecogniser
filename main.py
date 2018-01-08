import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from os import listdir
from os.path import isfile, join


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')

def segmentor(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        if len(eyes) >= 1 and len(mouth) >= 1:
        
            ex, ey, ew, eh = eyes[0]
            mx, my, mw, mh = mouth[0]
            
            cropped_mouth = gray[y + my:y + my + mh, x + mx:x + mx + mw]
            cropped_eyes = gray[y + ey:y + ey + eh, x + ex: x + ex + ew]

            cropped_mouth = cv2.resize(cropped_mouth, (50, 25))
            cropped_eyes = cv2.resize(cropped_eyes, (100, 25))

            return [cropped_eyes, cropped_mouth]
        else:
            return None

#====================
y = []
x_mouth = []
x_eyes = []
#   ADD JAFFE DATABASE FIRST
f = open('emotions_jaffe_lakshay.txt')
for line in f.read().split('\r\n'):
    y.append(int(line))

#   ADD 10K FACES NOW
f = open('emotions_10k_0_3388.txt')
for line in f.read().split('\r\n'):
    y.append(int(line))


f = open('emotions_10k_3389_6778.txt')
for line in f.read().split('\r\n'):
    y.append(int(line))


f = open('emotions_10k_6779_10167.txt')
for line in f.read().split('\r\n'):
    y.append(int(line))

y_labels = []
i = 0
#   COLLECT ALL JAFFE IMAGES
files = sorted([f for f in listdir('jaffe') if isfile(join('jaffe', f)) and 'tiff' in f])
for image in files:
    result = segmentor('jaffe/' + image)
    if result != None:
        x_mouth.append(result[1])
        x_eyes.append(result[0])
        y_labels.append(y[i])
    i += 1
#   COLLECT ALL 10K IMAGES
files = sorted([f for f in listdir('10kFaceImages') if isfile(join('10kFaceImages', f))])
for image in files:
    result = segmentor('10kFaceImages/' + image)
    if result != None:
        x_mouth.append(result[1])
        x_eyes.append(result[0])
        y_labels.append(y[i])
    i += 1

x_mouth = np.array(x_mouth)
x_eyes = np.array(x_eyes)
y_labels = np.array(y_labels)


#    Normalising the Data
x_mouth = x_mouth.astype('float32') / 128.0 - 1
x_eyes = x_eyes.astype('float32') / 128.0 - 1

x_mouth_train = x_mouth[:5574]
x_mouth_test = x_mouth[5574:]

x_eyes_train = x_eyes[:5574]
x_eyes_test = x_eyes[5574:]

y_labels_train = y_labels[:5574]
y_labels_test = y_labels[5574:]


def reformat(data, Y):
    xtrain = []
    trainLen = data.shape[0]
    for x in xrange(trainLen):
        xtrain.append(data[x,:,:])
    xtrain = np.asarray(xtrain)
    Ytr=[]
    for el in Y:
        temp=np.zeros(4)
        if el==1:
            temp[0]=1
        elif el==2:
            temp[1]=1
        elif el==3:
            temp[2]=1
        elif el==4:
            temp[3]=1
        Ytr.append(temp)
    return xtrain, np.asarray(Ytr)

mouth_train_data, mouth_train_labels = reformat(x_mouth_train, y_labels_train)
mouth_test_data, mouth_test_labels = reformat(x_mouth_test, y_labels_test)
eyes_train_data, eyes_train_labels = reformat(x_eyes_train, y_labels_train)
eyes_test_data, eyes_test_labels = reformat(x_eyes_test, y_labels_test)
mouth_train_data = np.reshape(mouth_train_data, (mouth_train_data.shape[0], mouth_train_data.shape[1], mouth_train_data.shape[2], 1))
mouth_test_data = np.reshape(mouth_test_data, (mouth_test_data.shape[0], mouth_test_data.shape[1], mouth_test_data.shape[2], 1))
eyes_train_data = np.reshape(eyes_train_data, (eyes_train_data.shape[0], eyes_train_data.shape[1], eyes_train_data.shape[2], 1))
eyes_test_data = np.reshape(eyes_test_data, (eyes_test_data.shape[0], eyes_test_data.shape[1], eyes_test_data.shape[2], 1))


#    Training CNN for mouth data
image_size = 25
width = 25
height = 50
height_eyes = 100
channels = 1

n_labels = 4
patch = 5
depth = 10
hidden = 64
dropout = 0.9375
batch = 10
learning_rate = 0.001


print "###########MOUTH###########"

tf_train_dataset = tf.placeholder(tf.float32, shape=(batch, width, height, channels))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch, 4))

tf_test_dataset = tf.constant(mouth_test_data)

layer1_weights = tf.Variable(tf.truncated_normal([patch, patch, channels, depth], stddev=0.1))
layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

layer2_weights = tf.Variable(tf.truncated_normal([patch, patch, depth, depth], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

layer3_weights = tf.Variable(tf.truncated_normal([910, 64], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[hidden]))

layer4_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))

dropout = tf.placeholder(tf.float32)
def model(data):
    conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)
    #   Max Pool
    hidden2 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #   Convolution 2 and RELU
    conv2 = tf.nn.conv2d(hidden2, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(conv2 + layer2_biases)
    #   Max Pool
    hidden4 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    shape = hidden4.get_shape().as_list()
    
    reshape = tf.reshape(hidden4, [-1, shape[1] * shape[2] * shape[3]])
    hidden5 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    #   Dropout
    dropout_layer = tf.nn.dropout(hidden5, 0.93)
    print hidden5.get_shape().as_list()
    print dropout_layer.get_shape().as_list()
    final_mat = tf.matmul(dropout_layer, layer4_weights) + layer4_biases
    print final_mat.get_shape().as_list()
    return final_mat
logits = model(tf_train_dataset)
print "Logits:", logits.get_shape().as_list()
print "Labels:", tf_train_labels.get_shape().as_list()
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

train_prediction = tf.nn.softmax(logits)
test_prediction = tf.nn.softmax(model(tf_test_dataset))

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
num_steps = 10000

with tf.Session() as session:
    print "TRAINING"
    tf.initialize_all_variables().run()
    average = 0
    for step in range(num_steps):
        #   Constucting the batch from the data set
        offset = (step * batch) % (mouth_train_labels.shape[0] - batch)
        batch_data = mouth_train_data[offset:(offset + batch), :, :]
        batch_labels = mouth_train_labels[offset:(offset + batch), :]
        #   Dictionary to be fed to TensorFlow Session
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, dropout: 0.93}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        #   Calculating the Accuracy of the predictions
        accu = accuracy(predictions, batch_labels)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accu)
        average += accu
    print "Average Accuracy : ", (average / num_steps)
    print "TESTING"
    print "Average Testing Accuracy : ", accuracy(test_prediction.eval(), mouth_test_labels)


print "###########EYES###########"

tf_train_dataset = tf.placeholder(tf.float32, shape=(batch, width, height_eyes, channels))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch, 4))

tf_test_dataset = tf.constant(eyes_test_data)

layer1_weights = tf.Variable(tf.truncated_normal([patch, patch, channels, depth], stddev=0.1))
layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

layer2_weights = tf.Variable(tf.truncated_normal([patch, patch, depth, depth], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

layer3_weights = tf.Variable(tf.truncated_normal([1750, 64], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[hidden]))

layer4_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))

dropout = tf.placeholder(tf.float32)
def model(data):
    conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)
    #   Max Pool
    hidden2 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #   Convolution 2 and RELU
    conv2 = tf.nn.conv2d(hidden2, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(conv2 + layer2_biases)
    #   Max Pool
    hidden4 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    shape = hidden4.get_shape().as_list()
    reshape = tf.reshape(hidden4, [-1, shape[1] * shape[2] * shape[3]])
    hidden5 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    #   Dropout
    dropout_layer = tf.nn.dropout(hidden5, 0.93)
    final_mat = tf.matmul(dropout_layer, layer4_weights) + layer4_biases
    return final_mat

logits = model(tf_train_dataset)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

train_prediction = tf.nn.softmax(logits)
test_prediction = tf.nn.softmax(model(tf_test_dataset))

with tf.Session() as session:
    tf.initialize_all_variables().run()
    print "TRAINING"
    average = 0
    for step in range(num_steps):
        #   Constucting the batch from the data set
        offset = (step * batch) % (eyes_train_labels.shape[0] - batch)
        batch_data = eyes_train_data[offset:(offset + batch), :, :]
        batch_labels = eyes_train_labels[offset:(offset + batch), :]
        #   Dictionary to be fed to TensorFlow Session
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, dropout: 0.93}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        #   Calculating the Accuracy of the predictions
        accu = accuracy(predictions, batch_labels)
        if (step % 100 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accu)
        average += accu
    print "Average Training Accuracy : ", (average / num_steps)
    print "Average Testing Accuracy: ", accuracy(test_prediction.eval(), eyes_test_labels)