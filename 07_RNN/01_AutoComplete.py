## 자연어 처리나 음성 처리 분야에 많이 사용되는 Recurrent Neural Network를 사용한다.
## RNN은 sequence가 있는 데이터에 사용된다.
## 4개 알파벳으로 이뤄진 단어를 학습시켜, 세글자만 주어지만 나머지 한글자를 추천하여 단어를 완성한다.

import tensorflow as tf
import numpy as np


char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']


# one-hot 인코딩 & 디코딩을 하기 위해 딕셔너리을 생성한다.
# {'a':0, 'b':1, ..., 'z':26}
num_dic = {n:i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)


## 다음 배열의 입력값과 출력값을 아래와 같이 사용하낟.
## wor -> X, d -> Y
## woo -> X, d -> Y
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']


## batch data를 만들기 위한 함수
def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        ## 여기서 생성하는 input_batch와 target_batch는 알파뱃 배열의 인덱스 번호이다.
        ## [22, 14, 17] [22, 14, 14], [3, 4, 4] ...
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]

        ## ont-hot 인코딩 처리를 한다.
        input_batch.append(np.eye(dic_len)[input])

        ## softmax_cross_entropy_with_logits()는 label 값을 one-hot 인코딩으로 전달해야 하지만,
        ## sparse_softmax_cross_entropy_with_logits()는 one-hot 인코딩을 사용하지 않으므로 index를 그냥 전달하면 된다.
        target_batch.append(target)

    return input_batch, target_batch


learning_rate = 0.01
n_hidden = 128
total_epoch = 30


## 타임스탭: [1 2 3] => 3
## RNN 을 구성하는 시퀀스 갯수
n_step = 3

## 입력값 크기, 알파벳에 대한 one-hot 인코딩이므로 26개가 된다.
## 예) c => [0 0 1 0 0 0 0 ..... 0 0]
## 출력값도 입력값과 마찬가지로 26개의 알파벳으로 분류한다.
n_input = n_class = dic_len


## 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, n_step, n_input])
## 비용함수에 sparse_softmax_cross_entropy_with_logits()를 사용하므로
## 출력값의 계산을 위한 원본값의 형태는 one-hot vector 형태가 아니라 인덱스 숫자를 그대로 사용한다.
## 때문에 아래와 같이 하나의 값만 갖는 1차원 배열을 입력으로 받는다.
## [3] [3] [15] [4]
## 만약, 기존처럼 one-hot 인코딩을 사용한다면 입력값의 형태는 [None, n_class] 이여야 한다.
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))


## RNN 셀을 생성한다.
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# overfitting 방지를 위해 dropout 사용
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1)

## 여러개의 셀을 조합해서 사용하기 위해 셀을 추가로 생성한다.
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

## 여러개의 셀을 조합한 RNN 셀을 생성한다.
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])


## tf.nn.dynamic_rnn(): RNN cell을 이용해 RNN을 생성한다.
## If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...], or a nested tuple of such elements.
## If time_major == True, this must be a Tensor of shape: [max_time, batch_size, ...], or a nested tuple of such elements.
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)


## 최종 결과는 one-hot 인코딩 형식으로 만든다.
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
print(outputs)
print(W)
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)


## 신경망 모델 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    input_batch, target_batch = make_batch(seq_data)
    for epoch in range(total_epoch):
        _, loss = sess.run([train, cost], feed_dict={X:input_batch, Y:target_batch})

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    print('최적화 완료!')


    ## 결과 확인
    prediction = tf.cast(tf.argmax(model, 1), tf.int32)
    prediction_check = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    input_batch, target_batch = make_batch(seq_data)

    predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X:input_batch, Y:target_batch})


    predict_words = []
    for idx, val in enumerate(seq_data):
        last_char = char_arr[predict[idx]]
        predict_words.append(val[:3] + last_char)

    print('\n=== 예측 결과 ===')
    print('입력값:', [w[:3] + ' ' for w in seq_data])
    print('예측값:', predict_words)
    print('정확도:', accuracy_val)

