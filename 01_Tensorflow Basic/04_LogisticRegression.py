## Logistic Regression
## 확률 모델 중 하나로, 독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법이다.
## 주로 분류, 예측을 위한 모델로서 사용된다.


import tensorflow as tf
tf.set_random_seed(777)

x_data = [
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 3],
    [5, 3],
    [6, 2]
]

y_data = [
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]


X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
print(X)
print(Y)

W = tf.Variable(tf.random_normal(shape=[2, 1], name='weight'))
b = tf.Variable(tf.random_normal(shape=[1], name='bias'))


# Logistic Regression 에서 사용되는 수식은 아래와 같으며 sigmoid 라고 부른다.
# H(X) = 1 / (1 + e^-W(transpose) * X)
# TensorFlow 에서는 tf.sigmoid() 함수로 위의 수식을 제공한다.
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)


cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


## 예측 값의 정확도를 계산하는 부분이다.
## 예측 값이 기준 값(0.5)를 넘는지 기준으로 분류를 진행한다.
# tf.cast(): Tensor의 타입을 변경한다.
predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))


# 세션을 실행하고 학습을 진행한다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print('Step: {0}\tcost: {1}'.format(step, cost_val))

    
    ## 예측값과 손실값, 정확도를 출력한다.
    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={X:x_data, Y:y_data})
    print('Hypothesis: {0}\t Predict: {1}\t accuracy: {2}'.format(h, c, a))
