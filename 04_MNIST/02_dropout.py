## 04_MNIST/01_mnist.py와 거의 흡사하고 DropOut 기법이 추가된 코드이다.
## DropOut은 Overfitting을 방지하기 위한 기법 중 하나로, 학습이 진행될 때
## 신경망에 구성된 랜덤으로 뉴런 중 일부를 비활성화여 가중치를 조정한다.
## 학습시에 활성화/비활성화 비율을 정해주어 일부만 활성화된 상태로 학습을 진행하며
## 실제 테스트를 진행할 때는 모든 뉴런을 활성화하고 테스트를 진행한다.
## 또한 예측된 결과와 실제 결과가 일치하는지 pyplot을 이용해 확인한다.


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


## TensorFlow에 기본 내장된 mnist 모듈을 사용해 데이터를 가져온다.
## 해당되는 경로에 데이터가 없을 경우 자동으로 데이터를 다운로드한다.
mnist = input_data.read_data_sets('/home/bhappy/DataSet/mnist/data/', one_hot=True)


## 신경망 모델 구성
# 입력값은 [배치크기, 픽셀수(특성값)] 으로 이뤄져있다.
# 이미지는 28*28 크기로, 다시말해 784개의 특성을 포함하고 있다.
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
train = tf.train.AdamOptimizer(0.001).minimize(cost)


## 신경망 모델 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(30):
        total_cost = 0

        for i in range(total_batch):
            ## TensorFlow의 mnist 모듈에서 기본제공하는 next_batch()를 통해 지정한 batch 크기만큼의 데이터를 가져온다
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([train, cost], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.8})
            total_cost += cost_val

        print("Epoch: {0}\tAvg: {1:.3f}".format(epoch+1, total_cost/total_batch))

    print("최적화 완료")


    ## 결과 확인
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0}))


    ## pyplot을 통해 결과를 이미지로 확인한다.
    labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})

    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(2, 5, i + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('%d' % np.argmax(labels[i]))
        subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)

    plt.show()