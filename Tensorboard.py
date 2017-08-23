import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('inputs'):
	    with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
		biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
		Wx_plus_b = tf.matmul(inputs,Weights) + biases
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs
	 
x_data = np.linspace(-1,1,300)[:,np.newaxis]  
noise = np.random.normal(0,0.05,x_data.shape)   
y_data = np.square(x_data) - 0.5  + noise
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32,[None,1],name='x_input')
	ys = tf.placeholder(tf.float32,[None,1],name='y_input')
# add hidden layer
layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

# add prediction, output layer
prediction =  add_layer(layer1,10,1,activation_function=None)

# loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                  reduction_indices=[1]))
# train step
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

# for i in range(1000):
    # sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    # if i % 50 == 0:
        # try:
            # ax.lines.remove(lines[0])
        # except Exception:
            # pass
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))     
        # prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # lines = ax.plot(x_data, prediction_value, 'b+', lw=10)
        # prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # ax.lines.remove(lines[0])
        # plt.pause(0.2)
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
