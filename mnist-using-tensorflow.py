import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
												#one is on and rest is of

'''
data set consists of digits 0-9, 10 classes
outputs:-
0 = [1,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0]...
'''		

n_nodes_hl1 = 500		
n_nodes_hl2 = 500		
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 
#take 100 entries manumpulate the weights and then input 100 more		

# height x width (24*24=785)
x = tf.placeholder('float',[None, 784]) #if data is not of this size tf gives you an error
y = tf.placeholder('float')

def neural_network_model(data):
	#generating a tensor with initial weights that will get modified
	hidden_layer_1 = {'weights':tf.Variable(tf.truncated_normal([784, n_nodes_hl1],stddev=0.1)),
					   'biases':tf.Variable(tf.constant(0.1,shape = [n_nodes_hl1]))}
	#Outputs random values from a normal distribution(tf.random_normal)
	hidden_layer_2 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
					    'biases':tf.Variable(tf.constant(0.1,shape = [n_nodes_hl2]))}
					   #input for hl2 is the number of nodes in the hl1, in this
	hidden_layer_3 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
					     'biases':tf.Variable(tf.constant(0.1,shape = [n_nodes_hl3]))}
					   
	output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
					     'biases':tf.Variable(tf.constant(0.1,shape = [n_classes]))}

	#(inputData*weights)+bias
	l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1) #activation function that tells you if a neuron has fired or not
 
	l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)  #input for layer 2 is activation function of l1

	l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])
	return output 				   				   				   




def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )#cost function
					#calculates the difference between the prediction and the lable that we have(one hot format)

	#now to minimize the cost (optimizer)
	optimizer = tf.train.AdamOptimizer().minimize(cost) #default learning rate is 0.001

	#cycles of feed forward and back-prop
	how_many_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		

		for epoch in range(how_many_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_,c = sess.run([optimizer, cost], feed_dict = {x:epoch_x,y:epoch_y})
				epoch_loss +=c
			print('epoch',epoch,'completed out of',how_many_epochs,'loss',epoch_loss)
			
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('accuracy',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)

'''
passing the data straight through (feed forward neural network)
input data -> weight it -> hiddenlayer 1(activation function) -> weight it
-> hiddenlayer 2(activation function) -> weight it -> output layer

(inputData*weights)+bias
	sometimes all the inputs are zero, then no nurons fire so you add a bias 

compare output to intended output
	you do it using the cost function(eg:- cross entropy)

then you use an optimiztion function
	to minimize the cost(eg:- AdamOptimizer...,sdg, AdaGrad)

what this optimization function does is, it goes back and changes the weights	
	back-propagation

1 cycle of feedForward and back-propagation is an epoch			

'''

