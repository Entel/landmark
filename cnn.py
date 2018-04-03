import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import csv

TRAINING_DATA_PATH = 'train.csv'
SIZE = (256, 256)
pointer = 0

def read_img_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img, True
    except:
        return 0, False

def load_data(path):
    data = []
    with open(path, 'rb') as f:
        spamreader = csv.reader(f, delimiter=',', quotechar=',')
        for row in spamreader:
            data.append(row)
    data = data[1:]
    return np.array(data)

def resize_image(img):
    width, height = img.size[0], img.size[1]
    if width > height:
        img = img.crop(((width-height)/2, 0, (width-height)/2+height, height))
        #bottom = Image.new('RGB', (width, width), 'white')
        #bottom.paste(img, (0, (width-height)/2))
    else:
        img = img.crop((0, (height-width)/2, width, (height-width)/2+width))
        #bottom = Image.new('RGB', (height, height), 'white')
        #bottom.paste(img, ((height-width)/2), 0)
        
    img = img.resize(SIZE, Image.ANTIALIAS)
    #print np.array(img.getdata()).shape
    try:
        img_data = np.reshape(list(img.getdata()), [256, 256, 3])
        return img_data, True
    except:
        return 0 ,False

def get_id_numbers(data):
    ids = []
    id_ = 0
    for row in data:
        new_id = int(row[2])
        if new_id not in ids:
            if new_id > id_:
                id_ = new_id
            ids.append(new_id)
    return np.array(ids).shape

data = load_data(TRAINING_DATA_PATH)
data_shape = data.shape[0]
output = get_id_numbers(data)[0]

def get_next_batch(batch_num=0, batch_size=10):
    batch_x = []
    batch_y = []
    global pointer
    y = np.zeros(output, dtype=int)
    for i in range(batch_size):
        batch_x.append(data[i][1])
        y[int(data[i][2])] = 1
        batch_y.append(list(y))
        pointer += 1
        if pointer > data_shape:
            pointer = 0
    return batch_x, batch_y

def data_process(x, y):
    batch_x = []
    batch_y = []
    i = 0
    for url in x:
        url = url.replace('"', '')
        img, t1 = read_img_from_url(url)
        if t1:
            img_data, t2 = resize_image(img)
            if t2:
                batch_x.append(img_data)
                batch_y.append(y[i])
        i += 1
    return batch_x, batch_y
        
        
'''
data = load_data(TRAINING_DATA_PATH)
print np.array(data).shape
print get_id_numbers(data)
img = read_img_from_url('http://static.panoramio.com/photos/original/70761397.jpg')
print img
resize_image(img).show()
'''
X = tf.placeholder('float', [None, 256, 256, 3])
Y = tf.placeholder('float', [None, output])
dropout_keep_prob = tf.placeholder(tf.float32)

def convulute_neural_network():
    weights = {'w_conv1': tf.Variable(tf.zeros([5, 5, 3, 16])),
                'w_conv2': tf.Variable(tf.zeros([5, 5, 16, 32])),
                'w_conv3': tf.Variable(tf.zeros([5, 5, 32, 64])),
                'w_conv4': tf.Variable(tf.zeros([5, 5, 64, 128])),
                'w_conv5': tf.Variable(tf.zeros([5, 5, 128, 256])),
                'w_conv6': tf.Variable(tf.zeros([3, 3, 256, 512])),
                'w_fc': tf.Variable(tf.zeros([4*4*512, 1024])),
                'w_out': tf.Variable(tf.zeros([1024, output]))}

    biases = {'b_conv1': tf.Variable(tf.zeros([16])),
                'b_conv2': tf.Variable(tf.zeros([32])),
                'b_conv3': tf.Variable(tf.zeros([64])),
                'b_conv4': tf.Variable(tf.zeros([128])),
                'b_conv5': tf.Variable(tf.zeros([256])),
                'b_conv6': tf.Variable(tf.zeros([512])),
                'b_fc': tf.Variable(tf.zeros([1024])),
                'b_out': tf.Variable(tf.zeros([output]))}

    conv1 = tf.nn.relu(tf.nn.conv2d(X, weights['w_conv1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv1'])
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv2'])
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights['w_conv3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv3'])
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights['w_conv4'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv4'])
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv5 = tf.nn.relu(tf.nn.conv2d(conv4, weights['w_conv5'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv5'])
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv6 = tf.nn.relu(tf.nn.conv2d(conv5, weights['w_conv6'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv6'])
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print tf.shape(conv1), tf.shape(conv2), tf.shape(conv3)
    print tf.shape(conv4), tf.shape(conv5), tf.shape(conv6)
    conv6_flat = tf.reshape(conv6, [-1, 4*4*512])
    fc = tf.nn.relu(tf.add(tf.matmul(conv6_flat, weights['w_fc']), biases['b_fc']))

    output_layer = tf.add(tf.matmul(fc, weights['w_out']), biases['b_out'])
    return output_layer

def train_cnn():
    predict = convulute_neural_network()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())
    accur = 0

    with open('loss.dat', 'w') as f:
        pass
    with open('accuracy.dat', 'w') as f:
        pass
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        times = 0

        while True:
            size = 64
            batch_x, batch_y = get_next_batch(pointer, size)
            batch_x_img, batch_y = data_process(batch_x, batch_y)

            #print np.array(batch_x_img).shape, np.array(batch_y).shape
            _, loss_ = sess.run([optimizer, loss], feed_dict={X:np.array(batch_x_img, dtype=np.uint8), Y:batch_y, dropout_keep_prob: 0.5})
            print(times,loss_, accur)
            with open('loss.dat', 'a+') as f:
                f.write(str(loss_) + '\n')
            times += 1
            
            if times % 100 == 0:
                test_x, test_y = get_next_batch(pointer,400)
                test_x_img, test_y = data_process(test_x, test_y)
                predictions = tf.argmax(predict, 1)
                correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
                accur = sess.run(accuracy, feed_dict={X:np.array(test_x_img, dtype=np.uint8), y:test_y, dropout_keep_prob:1.0})
                with open('accuracy.data', 'a+') as f:
                    f.write(str(accur) + '\n')
            if times % 500 == 0:
                saver.save(sess, '../tmp/landmark.model', global_step=times)

train_cnn()
            
