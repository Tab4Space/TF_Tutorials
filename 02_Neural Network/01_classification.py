## 털과 날개의 유무에 따라, 포유류인지 조류인지 분류하는 신경망 모델


import tensorflow as tf
import numpy as np


# [털, 날개]   0:없음, 1:있음
## 아래와 같은 데이터 형식을 one-hot 인코딩 데이터라고 한다.
## 표기하기 쉽고 명확하지만, 데이터의 양이 많아지면 표현하는데 무리가 있다.
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

# 들어오는 데이터는 차원과 알맞게 가충치의 차원을 정해준다.
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
# 편향(bias)은 각 레이어의 출력 개수로 정한다.
b = tf.Variable(tf.zeros([3]))


# 신경망에 가중치 W와 편향 b를 적용한다.
L = tf.add(tf.matmul(X, W), b)
# 가중치와 편향이 적용된 결과 값에 TensorFlow에서 적용하는 기본 Activation Function을 적용한다.
L = tf.nn.relu(L)


## softmax는 두 가지 이상 분류할 때, 사용하는 분류기법이다.
## 이진분류를 여러번 결합한 결과로 예측 결과가 카테고리A일 확률, B일 확률, C일 확률을 모두 구하여
## 그 중 가장 큰 확률을 가진 카테고리로 분류한다.
model = tf.nn.softmax(L)


## 신경망을 최적화하기 위한 cost function을 작성한다.
## 각 개별 결과에 대한 합을 얻고, 그 평균을 내는 방식을 사용한다.
## 전체 합이 아닌, 개별 결과를 구한 뒤 평균을 내는 방식을 사용하기 위해 axis 옵션을 사용한다.
## axis 옵션이 없으면 -1.09 처럼 총합인 스칼라값으로 출력된다..
##        Y         model         Y * tf.log(model)   reduce_sum(axis=1)
## 예) [[1 0 0]  [[0.1 0.7 0.2]  -> [[-1.0  0    0]  -> [-1.0, -0.09]
##     [0 1 0]]  [0.2 0.8 0.0]]     [ 0   -0.09 0]]
## 즉, 이것은 예측값과 실제값 사이의 확률 분포의 차이를 비용으로 계산한 것이며, 이것을 Cross-Entropy 라고 한다.
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# 신경망 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})

        print('Step: {0}\tCost: {1}'.format(step, cost_val))

        
    #########
    # 결과 확인
    # 0: 기타 1: 포유류, 2: 조류
    ######
    # tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 출력한다.
    # 예) [[0 1 0] [1 0 0]] -> [1 0] -> (1번째 인덱스, 0번째 인덱스)
    #    [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0] -> (1번째 인덱스, 0번째 인덱스)
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
    print('실제값:', sess.run(target, feed_dict={Y: y_data}))

    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))