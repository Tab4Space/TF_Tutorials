## Convolution Neural Network
## 합성곱 신경망이라고도 불리며 이미지 분야에서 많이 사용된다.
## Convolution + Pooling + Fully Connected 로 구성되어 있다.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# mnist 데이터 로드
mnist = input_data.read_data_sets("/home/bhappy/DataSet/mnist/data/", one_hot=True)


## 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

## CNN 에서는 filter(kernel이라고도 함)가 이미지를 훑어가며 연산을 수행한다.
## CNN 에서 사용되는 변수와 레이어는 아래와 같은 형태로 구성된다.
## W1 = [3, 3, 1, 32] => [3, 3]: 필터(커널)크기
##                       1: 입력 값(channel이란 개념으로 사용하는데, minist는 흑백이므로 값 1을 가짐)
##                       32: 출력 값(사용할 filter의 개수 개념)
## L1 Conv shape: (?, 28, 28, 32)
## L1 Pool shape: (?, 14, 14, 32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
## tf.nnconv2d(): TensorFlow에서 기본으로 제공하는 Conv Net 연산에 사용되는 함수이다.
## strides: 생성된 필터가 상하좌우 움직일 범위를 정할 수 있다.
## padding: 'SAME' 옵션을 지정하면 conv2d()가 적용될 때, 이미지 크기가 변경되지 않도록 할 수 있다.
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
## tf.nn.max_pool(): TensorFlow에서 기본으로 제공하는 Pooling 연산에 사용되는 함수이다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## L2 Conv shape: (?, 14 ,14, 64)
## L2 Pool shape: (?, 7, 7, 64)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## Fully Connected
## softmax 계산을 위해 다차원으로 구성된 특성값을 1차원으로 펴주는 작업
W3 = tf.Variable(tf.random_normal([7*7*64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7*7*64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

## 최종 출력 값
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
#train = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)


## 신경망 모델 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(15):
        total_cost = 0

        for i in range(total_batch):
            ## TensorFlow의 mnist 모듈에서 기본제공하는 next_batch()를 통해 지정한 batch 크기만큼의 데이터를 가져온다
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # placeholder에 맞게 shape을 변형한다.
            batch_x = batch_x.reshape(-1, 28, 28, 1)

            _, cost_val = sess.run([train, cost], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.7})
            total_cost += cost_val

        print("Epoch: {0}\tAvg: {1:.3f}".format(epoch+1, total_cost/total_batch))

    print("최적화 완료")


    ## 결과 확인
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, keep_prob:1.0}))
