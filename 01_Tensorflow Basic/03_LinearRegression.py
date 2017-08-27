## Linear Regression
## 변수들 사이의 관계를 분석하는데 사용하는 통계학적 방법이다.
## 이 방법의 장점은 알고리즘의 개념이 복잡하지 않고 다양한 문제에 적용할 수 있다는 것이다.

import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# tf.random_uniform(shape, minval, maxval, dtype.....)
# 지정한 범위의 균등분포를 생성한다. 
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


# name: 나중에 tensor board 등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 변수에 이름을 붙여준다.
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
print(X)
print(Y)


# X와 Y의 관계를 분석하기 위한 가설 수식을 정한다.
# y = W * x + b (일반적인 linear regression 수식)
# W와 X가 행렬이 아니므로 tf.matmul 대신, 기본 곱셉을 사용한다.
hypothesis = W * X + b


## 손실함수(cost function)
## 반복을 통해 W, b 값을 수정하며 정확한 결과를 얻기위해 노력한다. 
## 반복이 일어날 때마다 개선되고 있는지 확인하기 위해 얼마나 좋은 or 나쁜 예측인지 측정하는 것이 손실함수이다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# tensoflow에 기본적으로 포함된 함수를 이용해 경사하강법 최적화를 수행한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 손실을 최소화 하는것이 최종 목표이다.
train_op = optimizer.minimize(cost)


# 세션을 생성하고 초기화한다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        # sess.run()을 통해 train_op와 cost graph를 계산한다.
        # 수식에 들어갈 입력 값은 feed_dict를 통해 전달한다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인한다.
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))