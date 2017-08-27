import tensorflow as tf

# tf.placeholder: 계산에 필요한 입력 값을 외부? 에서 넣을 경우 입력 값을 받는 변수로 사용한다.
# None 은 크기(tensor의 크기)가 정해지지 않았음을 의미한다.
X = tf.placeholder(tf.float32, [None, 3])
print(X)


# tf.placeholder() 에서 크기를 정한 것과 같이 들어갈 값의 차원을 맞춰준다.
x_data = [[1, 2, 3], [4, 5, 6]]


# tf.Variable(): 변수를 생성한다. 보통 weight 값 등 최적화를 위한 변수에 사용된다.
# tf.random_normal(): 각 변수들의 초기 값을 정규분포 랜덤 값으로 초기화 한다.
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))


# 입력 값과 변수들을 계산할 수식을 정의한다.
# tf.matmul(): 행렬 계산을 수행한다.
expr = tf.matmul(X, W) + b


sess =  tf.Session()
# 위에서 설정한 tf.Variable() 변수들의 초기화를 위해 tf.global_variables_initializer()를 한 번 실행해야 한다.
sess.run(tf.global_variables_initializer())

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
# expr 수식에는 X라는 입력 값이 필요하다. 따라서, expr 실행 시에는 이 변수에 대한 실제 입력 값을 feed_dict를 이용해 넣어준다.
print(sess.run(expr, feed_dict={X:x_data}))


sess.close()