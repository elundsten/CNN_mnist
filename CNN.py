import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
sess = tf.Session()

#Layer one
filter_size1 = 5
num_filters1 = 16

#Layer two
filter_size2 = 5
num_filters2 = 36

#Layer four
filter_size3 = 5
num_filters3 = 106

#Layer three
filter_size4 = 5
num_filters4 = 156
# FC-layer
fc_size = 128 

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
y_true = tf.one_hot(y_train,10)
y_test_true = tf.one_hot(y_test,10)
y_true_cls = y_train
x_train_flat = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
#Dimensions
img_size_flat = x_train_flat.shape[1]
img_size = x_train.shape[1]
img_shape = (x_train.shape[1],x_train.shape[2])
num_test = x_test.shape[0]
print("numtest:" ,num_test)
num_classes = 10
num_channels = 1
#Input and output data

#Weights and biases
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

#run it
sess.run(tf.global_variables_initializer())
y_one_hot = sess.run(y_true)
y_test_true = sess.run(y_test_true)
# Get the true classes for those images.
images = x_train[0:9,:,:]
cls_true = y_true_cls[0:9]
def pooling(layer):
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
            return layer
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   poooling_active=True):  # Use 2x2 max-pooling.


    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)

    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases
    if poooling_active:
        layer = pooling(layer)

    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    print(num_features)

    layer_flat = tf.reshape(layer, [-1, num_features])
    print(layer_flat.shape)

    return layer_flat, num_features
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   poooling_active=True)

print(layer_conv1.shape)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   poooling_active=True)
layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   poooling_active=True)
layer_conv4, weights_conv4 = \
    new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters3,
                   filter_size=filter_size4,
                   num_filters=num_filters4,
                   poooling_active=True)

print(layer_conv2.shape)

layer_flat, num_features = flatten_layer(layer_conv4)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

print(layer_fc1)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

train_batch_size = 64
total_iterations = 0

def optimize(num_iterations):
	# Ensure we update the global variable rather than a local copy.
	global total_iterations

	# Start-time used for printing time-usage below.
	start_time = time.time()
	for i in range(total_iterations,total_iterations + num_iterations):

		if ((i+1)*train_batch_size > 60000):
			start = i*train_batch_size - 60000
			end = (i+1)*train_batch_size - 60000
			x_batch = x_train_flat[start:end,:]
			y_true_batch = y_one_hot[start:end,:]
		else:
			x_batch = x_train_flat[i*train_batch_size:(i+1)*train_batch_size,:]
			y_true_batch = y_one_hot[i*train_batch_size:(i+1)*train_batch_size,:]

		feed_dict_train = {x: x_batch,
		                   y_true: y_true_batch}

		sess.run(optimizer, feed_dict=feed_dict_train)

		if i % 100 == 0:
			acc = sess.run(accuracy, feed_dict=feed_dict_train)

			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

			print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
	total_iterations += num_iterations

    # Ending time.
	end_time = time.time()

	# Difference between start and end-times.
	time_dif = end_time - start_time

	# Print the time-usage.
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):

    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + train_batch_size, num_test)

        images = x_test[i:j, :]

        labels = y_test_true[i:j, :]

        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = y_test

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
optimize(1000)
print_test_accuracy()