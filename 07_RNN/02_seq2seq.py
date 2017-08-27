## seq2seq는 챗봇, 번역, 이미지 캡셔닝 등에 사용되는 시퀀스 학습/생성 모딜이다.
## 영어 단어를 한국어 단어로 번역한다.

import tensorflow as tf
import numpy as np

## S: 디코딩 입력의 시작을 나타내는 심볼
## E: 디코딩 출력을 끝을 나타내는 심볼
## P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
##    예) 현재 배치 데이터의 최대 크기가 4 인 경우
##       word -> ['w', 'o', 'r', 'd']
##       to   -> ['t', 'o', 'P', 'P']


char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
print(char_arr)
num_dic = {n:i for i,n in enumerate(char_arr)}
print(num_dic)
dic_len = len(num_dic)


## 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [
    ['word', '단어'], ['wood', '나무'],
    ['game', '놀이'], ['girl', '소녀'],
    ['kiss', '키스'], ['love', '사랑']
]


## 배치 데이터 생성
def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값, 입력다어의 글자들을 한글자씩 떼어 배열로 만든다.
        input = [num_dic[n] for n in seq[0]]
        # 디코더 셀의 입력값, 시작을 나타내는 S 심볼을 맨 앞에 붙인다.
        output = [num_dic[n] for n in ('S'+seq[1])]
        # 학습을 위해 비교할 디코더 셀의 출력값, 끝나는 것을 알려주기 위해 마지막 E 심볼을 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        # print(input)
        # print(output)
        # print(target)

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    # print()
    # print(input_batch)
    # print(output_batch)
    # print(target_batch)
    
    return input_batch, output_batch, target_batch


learning_rate = 0.01
n_hidden = 128
total_epoch = 100
n_class = n_input = dic_len


## 신경망 모델 구성
## seq2seq 모델은 인코더 입력과 디코더의 입력 형식이 같다.
enc_input = tf.placeholder(tf.float32, [None, 4, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])        # [batch size, time steps]


## 인코더 셀을 구성
with tf.variable_scope('encoder'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

# 디코더 셀을 구성
with tf.variable_scope('decoder'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    ## seq2seq 모델은 인코더 셀의 최종 상태값을 디코더 셀의 초기 상태값으로 넣어주는 것이 중요하다.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)


model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)


## 신경망 모델 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    input_batch, output_batch, target_batch = make_batch(seq_data)
    # print(input_batch)
    # print()
    # print(output_batch)
    # print()
    # print(target_batch)
    for epoch in range(total_epoch):
        _, cost_val = sess.run([train, cost], feed_dict={enc_input:input_batch, dec_input:output_batch, targets:target_batch})

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(cost_val))
        
    print('최적화 완료!')


    ## 번역 테스트
    # 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
    def translate(word):
        # 작성한 모델은 입력값과 출력값 데이터로 [영단어, 국문단어]를 사용하지만, 예측시에는 한글단어를 모른다.
        # 따라서 디코더의 입출력값을 의미 없는 값인 'P'로 채운다

        seq_data = [word, 'P'*len(word)]
        input_batch, output_batch, target_batch = make_batch([seq_data])

        # 결과가 [batch size, time step, input] 으로 나오기 때문에 2번째 차원인 input 차원을 argmax로 가장 확률이 높은 글자를 예측값으로 만든다.
        prediction = tf.argmax(model, 2)
        result = sess.run(prediction, feed_dict={enc_input:input_batch, dec_input:output_batch, targets:target_batch})

        # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
        decoded = [char_arr[i] for i in result[0]]

        # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated


    print('\n=== 번역 테스트 ===')

    print('word ->', translate('word'))
    print('wodr ->', translate('wodr'))
    print('love ->', translate('love'))
    print('loev ->', translate('loev'))
    print('abcd ->', translate('abcd'))
