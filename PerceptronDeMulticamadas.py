import tensorflow as tf
from tqdm import trange
from tensorflow.examples.tutorials.mnist import input_data

# Importa os dados
mnist = input_data.read_data_sets("datasets/MNIST_data/", one_hot=True)

# Cria o modelo
x = tf.placeholder(tf.float32, [None, 784])

# Define a arquitetura
hidden_units_1 = 500
hidden_units_2 = 100

# Camada 1
W1 = tf.Variable(tf.truncated_normal([784, hidden_units_1], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_units_1]))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# Camada 2
W2 = tf.Variable(tf.truncated_normal([hidden_units_1, hidden_units_2], stddev=0.1))
b2 = tf.Variable(tf.zeros([hidden_units_2]))
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

# Camada de saída
W_out = tf.Variable(tf.truncated_normal([hidden_units_2, 10], stddev=0.1))
b_out = tf.Variable(tf.zeros([10]))
y = tf.matmul(h2, W_out) + b_out

# Definir perda e otimizador
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Criar um objeto Session, inicializar todas as variáveis

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Treinamento
for _ in trange(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Teste do treinamento do modelo
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Test accuracy: {0}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

# Fechar a sessão
sess.close()