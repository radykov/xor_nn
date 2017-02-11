import tensorflow as tf

sess = tf.InteractiveSession()

# define placeholder for input, None as first argument means tensor can be any length
x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

W = tf.Variable(tf.random_uniform([2, 2], -.01, .01))
b = tf.Variable(tf.random_uniform([2], -.01, .01))
hidden  = tf.nn.relu(tf.matmul(x_,W) + b) # first layer.

W2 = tf.Variable(tf.random_uniform([2,1], -.1, .1))
b2 = tf.Variable(tf.zeros([1]))
hidden2 = tf.matmul(hidden, W2 + b2)

y = tf.nn.softmax(hidden2)
cross_entropy = -tf.reduce_sum(y_*tf.log(hidden2))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess.run(init)

# Train on the input data
for i in range(100):
    sess.run([cross_entropy, train_step], feed_dict={x_: XOR_X, y_: XOR_Y})
    print ('W1', sess.run(W))
    print('Output ', sess.run(y, feed_dict={x_: XOR_X, y_: XOR_Y}))

