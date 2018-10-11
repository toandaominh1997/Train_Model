# @Author: bigkizd(Toan Dao Minh)
# @ Model using CNN raw to train HVHN datasets

import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf 

train_data = sio.loadmat('train_32x32.mat')
X_train = train_data['X']
y_train = train_data['y']

Yt = np.zeros((y_train.shape[0], 10))
for i in range(0, y_train.shape[0]):
    Yt[i][y_train[i][0]%10]=1
#    print(y_train[i][0])
X = np.zeros((73257, 32, 32, 3))
for i in range(0, X_train.shape[3]):
    X[i]=X_train[:, :, :, i]



# Parameters
num_input = 32*32*3
num_classes = 10
dropout = 0.75
num_steps = 1000
batch_size = 128
learning_rate = 0.001
num_steps = 200
display_step = 10
weights ={
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}
biases={
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes])),
}
def conv2d(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpooling2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
def conv_net(x, weights, biases, dropout):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpooling2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpooling2d(conv2, k=2)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
_X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
logits = conv_net(_X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1, num_steps+1):
        begin = i*batch_size
        end = begin+batch_size
        if(end>=X.shape[0]):
            end = X.shape[0]
        train_x, train_y = X[begin:end, :, :, :], Yt[begin:end, :]
        sess.run(train_op, feed_dict={_X:train_x, Y: train_y, keep_prob: 0.8})
        if i % display_step == 0 or i == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={_X: train_x,
                                                                 Y: train_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(i) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
