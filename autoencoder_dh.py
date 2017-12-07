# 대표적인 비지도(Unsupervised) 학습 방법인 Autoencoder 를 구현해봅니다.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


##############
# 파일 로드
##############
def load_data_file(fname, num_flds=0):
  data = []
  with open(fname) as fp:
    next(fp) # skip the first line
    for line in fp:
      flds = line.split()
      if num_flds and len(flds) != num_flds:
        #print('ODD',  fname, len(flds), line[:-1])
        continue
      v = np.array(flds[1:], dtype=np.float32)
      #print(v)
      data.append(v)
      #break
  # np.matrix() 호출시에 data 내의 각 원소들의 차수가 맞지 않으면 오류가 발생한다.
  # 그래서 num_flds를 확인하는 것이 중요하다.
  return np.matrix(data=data, dtype=np.float32, copy=False)

def load_data_file2(fname, num_flds=0,num=1):
  data = []
  v=np.zeros(num*10)
  count=0
  linenum=0
  with open(fname) as fp:
    linenum=sum(1 for line in open(fname))
    #print('줄수',linenum)
    next(fp) # skip the first line
    for line in fp:
      flds = line.split()
      if num_flds and len(flds) != num_flds:
        #print('ODD',  fname, len(flds), line[:-1])
        continue
      #v = np.array(flds[1:], dtype=np.float32)
      '''
      for i in range(0, 9):
          v[count] = flds[i + 1]
          count = count + 1
      '''
      if count + 77 < linenum:
        for i in range(0,10):
            v[10*(count%77)+i]=flds[i+1]
        if count%76==0:
              if count!=0:
                #print('몇에서 append?',count)
                #print(v)
                data.append(v)
      count = count + 1
      #break
  # np.matrix() 호출시에 data 내의 각 원소들의 차수가 맞지 않으면 오류가 발생한다.
  # 그래서 num_flds를 확인하는 것이 중요하다.
  return np.matrix(data=data, dtype=np.float32, copy=False)



degreedata = load_data_file2('DegreeVector_8lrKv75pdy4.txt', num_flds=10+1,num=77)
print('DEG', degreedata.shape)
#print(degreedata[0,0])
#print(degreedata[0,1])






#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#########
# 옵션 설정
######
learning_rate = 0.01
training_epoch = 100
batch_size = 100
# 신경망 레이어 구성 옵션
n_hidden = 10  # 히든 레이어의 뉴런 갯수
n_input = 770   # 입력값 크기 - 이미지 픽셀수

#########
# 신경망 모델 구성
######
# Y 가 없습니다. 입력값을 Y로 사용하기 때문입니다.
X = tf.placeholder(tf.float32, [None, n_input])

# 인코더 레이어와 디코더 레이어의 가중치와 편향 변수를 설정합니다.
# 다음과 같이 이어지는 레이어를 구성하기 위한 값들 입니다.
# input -> encode -> decode -> output
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
# sigmoid 함수를 이용해 신경망 레이어를 구성합니다.
# sigmoid(X * W + b)
# 인코더 레이어 구성
encoder = tf.nn.tanh(
                tf.add(tf.matmul(X, W_encode), b_encode))

# encode 의 아웃풋 크기를 입력값보다 작은 크기로 만들어 정보를 압축하여 특성을 뽑아내고,
# decode 의 출력을 입력값과 동일한 크기를 갖도록하여 입력과 똑같은 아웃풋을 만들어 내도록 합니다.
# 히든 레이어의 구성과 특성치을 뽑아내는 알고리즘을 변경하여 다양한 오토인코더를 만들 수 있습니다.
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
# 디코더 레이어 구성
# 이 디코더가 최종 모델이 됩니다.
decoder = tf.nn.tanh(
                tf.add(tf.matmul(encoder, W_decode), b_decode))

# 디코더는 인풋과 최대한 같은 결과를 내야 하므로, 디코딩한 결과를 평가하기 위해
# 입력 값인 X 값을 평가를 위한 실측 결과 값으로하여 decoder 와의 차이를 손실값으로 설정합니다.
cost = tf.reduce_mean(tf.pow(X - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#########
# 신경망 모델 학습
######
global_step = tf.Variable(0,trainable=False, name='global_step')
init = tf.global_variables_initializer()
sess = tf.Session()
saver=tf.train.Saver(tf.global_variables())
ckpt=tf.train.get_checkpoint_state('./model')
# if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#     saver.restore(sess, ckpt.model_checkpoint_path)
# else:
#     sess.run(init)
sess.run(init)

total_batch = int(4000/batch_size)
#mergedata=[]
#print (degreedata.size/10)
#for i in range(degreedata.size):
#   if i%2==0:
#        for j in range(0,20):
#            k=int(i/2)
#            print('k',k)
#            if j<10:
#                mergedata[(k,j)]=degreedata[(i,j)]
#            else:
#                mergedata[(k,j)]=degreedata[(i,j-10)]

for epoch in range(training_epoch):
    total_cost = 0
    #print(encoder)

    for i in range(int(total_batch)):
        batch_xs = degreedata
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

print('최적화 완료!')

saver.save(sess,'./model/dnn.ckpt',global_step=global_step)
#########
# 결과 확인
# 입력값(위쪽)과 모델이 생성한 값(아래쪽)을 시각적으로 비교해봅니다.
######

sample_size = 10

#samples = sess.run(decoder,
#                   feed_dict={X: mnist.test.images[:sample_size]})
samples = sess.run(encoder,
                   feed_dict={X: degreedata[:sample_size]})

for i in range(sample_size):
    #print("in")
    #print(degreedata[i])
    print('[%02d]' % i, samples[i])
    pass
    

#fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

#for i in range(sample_size):
#    ax[0][i].set_axis_off()
#    ax[1][i].set_axis_off()
#    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
#    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

#plt.show()

