## 02_Neural Network/01_classification.py와 같은 데이터를 사용한다.
## 단, 하드코딩된 상태가 아닌 csv 파일에 담긴 데이터를 이용한다.


import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')
# 데이터의 차원을 맞추기 위해 전치를 적용한다.
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
print(x_data)
print(y_data)


## 신경망 모델 구성
# 학습에 영향이 미치는 변수가 아닌 학습 횟수에 따라 단순히 증가시킬 변수이다.
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 20], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([20, 3], -1.0, 1.0))

L1 = tf.nn.relu(tf.matmul(X, W1))
L2 = tf.nn.relu(tf.matmul(L1, W2))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


## 신경망 모델 학습
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # 최적화 진행
    for step in range(2):
        _, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})

        print('Step: {0}, Cost: {1}'.format(global_step, cost_val))

    
    saver.save(sess, './model/dnn.ckpt', global_step=global_step)


    ## 결과 확인
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
    print('실제값:', sess.run(target, feed_dict={Y: y_data}))

    check_prediction = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))