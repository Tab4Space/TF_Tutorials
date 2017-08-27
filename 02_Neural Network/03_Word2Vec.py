import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


## matplotlib에서 한글을 표시하기 위한 설정
font_name = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()
matplotlib.rc('font', family=font_name)


# 단어 벡터를 분석해볼 임의의 문장들
sentences = ["나 고양이 좋다",
             "나 강아지 좋다",
             "나 동물 좋다",
             "강아지 고양이 동물",
             "여자친구 고양이 강아지 좋다",
             "고양이 생선 우유 좋다",
             "강아지 생선 싫다 우유 좋다",
             "강아지 고양이 눈 좋다",
             "나 여자친구 좋다",
             "여자친구 나 싫다",
             "여자친구 나 영화 책 음악 좋다",
             "나 게임 만화 애니 좋다",
             "고양이 강아지 싫다",
             "강아지 고양이 좋다"]


# 문장을 전부 합친 후 공백으로 단어를 분리하고 고유한 단어들로 리스트를 만든다.
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
print(word_list)


## 문자열 자체로 분석을 하기보다는, 숫자로 분석하는 것이 편리하다.
## 따라서 인덱스와 문자로 이루어진 딕셔너리를 만든다.
## 이때, key(문자):value(인덱스) 구조로 딕셔너리가 만들어진다.
word_dict = {w: i for i, w in enumerate(word_list)}
word_index = [word_dict[word] for word in word_list]
print(word_dict)
print(word_index)


## 윈도우 사이즈를 1로 하는 skip-gram 모델을 생성한다.
# 예) 나 게임 만화 애니 좋다
#   -> ([나, 만화], 게임), ([게임, 애니], 만화), ([만화, 좋다], 애니)
#   -> (게임, 나), (게임, 만화), (만화, 게임), (만화, 애니), (애니, 만화), (애니, 좋다)
skip_grams = []

for i in range(1, len(word_index) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    target = word_index[i]
    context = [word_index[i - 1], word_index[i + 1]]

    # (target, context[0]), (target, context[1])..
    for w in context:
        skip_grams.append([target, w])


## skip-gram 데이터에서 랜덤으로 배치 데이터를 생성한다.
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels


## 학습에 필요한 hyper parameter를 정한다.
training_epoch = 300
learning_rate = 0.1
batch_size = 20
embedding_size = 2          # 단어 벡터를 구성할 임베딩 차원의 크기
num_sampled = 15            # nce_loss 함수에서 사용하기 위한 샘플링 크기이며, batch_size 보다 작아야한다.
voc_size = len(word_list)


## 신경망 모델 구성
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])


## word2vec 모델의 결과 값인 embedding vector를 저장한다.
## 총 단어 갯수와 embedding 개수를 크기로 하는 2차원을 갖는다.
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))

## 임베딩 벡터의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다.
## 예) embeddings     inputs    selected
##    [[1, 2, 3]  -> [2, 3] -> [[2, 3, 4]
##     [2, 3, 4]                [3, 4, 5]]
##     [3, 4, 5]
##     [4, 5, 6]]
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)


## nce_loss 함수에서 사용할 변수
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))


## nce loss는 TensorFlow에서 기본 함수로 제공한다.
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


## 신경망 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        if step % 10 == 0:
            print("loss at step ", step, ": ", loss_val)

    # matplot 으로 출력하여 시각적으로 확인해보기 위해 임베딩 벡터의 결과 값을 계산하여 저장한다.
    # with 구문 안에서는 sess.run 대신 간단히 eval() 함수를 사용할 수 있다.
    trained_embeddings = embeddings.eval()


## 임베딩된 Word2Vec 결과 확인
## 결과는 해당 단어들이 얼마나 다른 단어와 인접해 있는지를 보여준다.
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.show()