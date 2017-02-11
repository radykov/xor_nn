import tensorflow as tf

sess = tf.InteractiveSession()

# define placeholder for input and output
x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

# Configure weights and layers. 10 hidden layer neurons helps prevent susceptibility to randomness
# Since we're using a relu, the weights are initiated appropriately to avoid dead (-ve) neurons
W = tf.Variable(tf.random_uniform([2, 10], 0.001, .01))
b = tf.Variable(tf.zeros([10]))
hidden  = tf.nn.relu(tf.matmul(x_,W) + b)

W2 = tf.Variable(tf.random_uniform([10,1], -1, 1))
b2 = tf.Variable(tf.zeros([1]))
hidden2 = tf.matmul(hidden, W2) + b2
y = tf.nn.sigmoid(hidden2)

# Training function allows for error calculations for value between 0 and 1
cost = tf.reduce_mean(( (y_ * tf.log(y)) +
	((1 - y_) * tf.log(1.0 - y)) ) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Specify the data to go into the placeholders
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess.run(init)

# Train on the input data, doesn't actually need 100000 to converge
for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
    # Print progress every so by examining the outputs and check how the weights are changing
    if i % 2000 == 0:
        print ('W1', sess.run(W))
        print('Output ', sess.run(y, feed_dict={x_: XOR_X, y_: XOR_Y}))

