

import tensorflow as tf
import numpy as np
import glob



def load_data_file(fname, num_flds=0):
  data = []
  with open(fname) as fp:
    next(fp) # skip the first line
    for line in fp:
      flds = line.split()
      if num_flds and len(flds) != num_flds:
        print('ODD',  fname, len(flds), line[:-1])
        continue
      v = np.array(flds[1:], dtype=np.float32)
      #print(len(v))
      data.append(v)
      #break

  return np.matrix(data=data, dtype=np.float32, copy=False)



def read_multiple_files_Degree():
    read_files = glob.glob("C:/Users/YeoChunghyun/Desktop/txttest/Degree/*.txt")
    with open("result_Degree.txt", "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                next(infile)
                outfile.write(infile.read())

def read_multiple_files_Position():
    read_files = glob.glob("C:/Users/YeoChunghyun/Desktop/txttest/Degree/*.txt")
    with open("result_Position.txt", "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                next(infile)
                outfile.write(infile.read())

## need seperate method
#def seperate_by_dirpath():


#degreedata = load_data_file('DegreeVector_8lrKv75pdy4.txt', num_flds=10+1)
#print('DEG', degreedata.shape)
#print(degreedata[1])
read_multiple_files_Degree()
read_multiple_files_Position()

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


"""
#########
# option
######
learning_rate = 0.01
training_epoch = 100
batch_size = 100
# neural layer 
n_hidden = 3  # hidden layer neuron mount
n_input = 10   # input scale - image pixel count

#########
# neural network model compose
######

X = tf.placeholder(tf.float32, [None, n_input])

# 
# 
# input -> encode -> decode -> output
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))

# sigmoid(X * W + b)
# 
encoder = tf.nn.tanh(
                tf.add(tf.matmul(X, W_encode), b_encode))


W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))

decoder = tf.nn.tanh(
                tf.add(tf.matmul(encoder, W_decode), b_decode))


cost = tf.reduce_mean(tf.pow(X - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#########

######
global_step = tf.Variable(0,trainable=False, name='global_step')
init = tf.global_variables_initializer()
sess = tf.Session()
saver=tf.train.Saver(tf.global_variables())
ckpt=tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_ceckpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(init)


total_batch = int(4000/batch_size)

for epoch in range(training_epoch):
    total_cost = 0
    print(encoder)

    for i in range(total_batch):
        batch_xs = degreedata[i]
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

print('optimization complete!')

saver.save(sess,'./model/dnn.ckpt',global_step=global_step)
#########

######

sample_size = 10

#samples = sess.run(decoder,
#                   feed_dict={X: mnist.test.images[:sample_size]})
samples = sess.run(decoder,
                   feed_dict={X: degreedata[:sample_size]})

for i in range(sample_size):
    print("in")
    print(degreedata[i])
    print(samples[i])

#fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

#for i in range(sample_size):
#    ax[0][i].set_axis_off()
#    ax[1][i].set_axis_off()
#    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
#    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

#plt.show()
"""



