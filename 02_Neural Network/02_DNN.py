## 02_Neural Network/01_classification.py와 같은 데이터를 사용한다.
## 레이어를 늘인 Deep Neural Network를 구성한다.


import tensorflow as tf
import numpy as np


x_data = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
])

y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])


# 신경망 모델 구성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# W1의 차원은 [특성개수, 히든레이어의 뉴런 개수]로 정한다.
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
# W2의 차원은 [히든레이어의 뉴런 개수, 분류개수]로 정한다.
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))
# bias는 각 가장치의 출력개수로 정한다.
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))


# 신경망의 히든 레이어에 W1과 b1을 적용한다.
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)
# 출력 레이어를 계산한다.
model = tf.add(tf.matmul(L1, W2), b2)


## 02_Neural Network/01_classification.py 에서는 cross-entropy 수식을 작성하였지만,
## Tensorflow에서 기본적으로 제공하는 cross-entropy 함수가 있다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
# GradientDescent처럼 최적화를 위한 함수이다.
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 신경망 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(100):
        _, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})
        print('Step: {0}\tCost: {1}'.format(step, cost_val))
    
    
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
    print('실제값:', sess.run(target, feed_dict={Y: y_data}))
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))