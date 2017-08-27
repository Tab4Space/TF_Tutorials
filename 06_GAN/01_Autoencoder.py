import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/bhappy/DataSet/mnist/data/", one_hot=True)


learning_rate = 0.01
training_epoch = 20
batch_size = 100
n_hidden = 256
n_input = 28*28


## 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, n_input])

# 인코더 레이어와 디코드 레이어 weight, bias
# input -> encode -> decode -> output
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))

# sigmoid 함수를 이용해 신경망 레이어를 구성한다.
# sigmoid(X * W + b)
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))


## encode 의 출력을 입력 값보다 작은 크기로 만들어 정보를 압축하여 특성을 추출하고, 
## decode 의 출력을 입력 값과 동일한 크기를 갖도록하여 입력과 똑같은 출력값을 만들어 내도록 한다.
## hidden layer의 구성과 특성치를 추출하는 알고리즈믕ㄹ 변경하면 다양한 AutoEncoder를 만들 수 있다.
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

## Decoder 는 input과 최대한 같은 결과를 내야 하므로, 디코딩한 결과를 평가하기 위해
## 입력 값인 X 값을 평가를 위한 실측 결과 값으로하여 Decoder 와의 차이를 손실 값으로 설정한다.
cost = tf.reduce_mean(tf.pow(X - decoder, 2))
train = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


## 신경망 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)

    for epoch in range(training_epoch):
        total_cost = 0
        
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={X:batch_x})

            total_cost += cost_val

        print("Epoch: {0}\tAvg: {1:.4f}".format(epoch+1, total_cost/total_batch))

    print('최적화 완료')


    ## pyplot을 이용해 생성된 값을 시각적으로 비교
    sample_size = 10
    samples = sess.run(decoder, feed_dict={X:mnist.test.images[:sample_size]})

    fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

    for i in range(sample_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

    plt.show()