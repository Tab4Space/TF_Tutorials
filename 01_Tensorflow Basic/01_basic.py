import tensorflow as tf

# tf.constant: constant(상수) tensor를 생성한다. 변하지 않는 tensor
hello = tf.constant('Hello, TensorFlow')
print(hello)

a = tf.constant(10)
b = tf.constant(20)
c = tf.add(a, b)
print(c)


## 위에서 상수, 변수, 수식을 정의하고 실행한다 하더라도 연산이 실행되는 것은 아니다.
## 아래와 같이 tf.Session() 클래스를 생성하고 run() 메소드를 사용할 때, 연산이 수행된다.
## 따라서 모델을 구성하는 것과, 실행하는 것을 분리하여 프로그램(애플리케이션)을 작성할 수 있다.

# 그래프를 실행할 세션을 구성한다.
# sess.run(): 설정한 tensor graph(변수, 수식 등)를 실행한다. 인자를 한 개 또는 리스트 형태로 여러 개를 넣을 수 있다.
sess = tf.Session()
print(sess.run(hello))
print(sess.run([a, b, c]))

sess.close()