## TensorFlow의 시각화 도구인 TensorBoard를 사용한다.
## 03_TensorBoard, Saver/01_saver.py와 거의 동일한 코드를 사용한다.


import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')
# 데이터의 차원을 맞추기 위해 전치를 적용한다.
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


## 신경망 모델 구성
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


## tf.name_scope(): 주어진 값이 동일한 그래프(모델)에서 왔는지 확인하고 그 그래프를
## 기본 그래프로 만들고 해당 그래프에서 이름 범위를 넣는다.
## with tf.name_scope() 로 묶은 블럭은 TensorBoard에서 한 레이어 안에 표현해준다.
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1.0, 1.0))
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1.0, 1.0))
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost, global_step=global_step)

    # tf.summary.scalar()를 이용해 수집하고 싶은 값들을 설정할 수 있다.
    tf.summary.scalar('cost', cost)


## 신경망 모델 학습
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    
    # TensorBoard에 표시하기 위한 Tensor들을 수집한다.
    merged = tf.summary.merge_all()
    # 저장할 그래프와 Tensor값들을 저장할 디렉토리를 설정한다.
    writer = tf.summary.FileWriter('./logs', sess.graph)
    ## 위와 같이 로그를 지정하면, 학습 이후에 아래 명령어를 이용해 웹서버를 실행시킨 뒤, TensorBoard를 확인할 수 있다.
    ## tensorboard --logdir=./logs  => 웹에서 http://localhost:6006 접속


    ## 최적화 진행
    for step in range(100):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        print('Step: %d, ' % sess.run(global_step),
            'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

        # 적절한 시점에 저장할 값들을 수집하고 저장합니다.
        summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=sess.run(global_step))

    saver.save(sess, './model/dnn.ckpt', global_step=global_step)


    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
    print('실제값:', sess.run(target, feed_dict={Y: y_data}))

    check_prediction = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))