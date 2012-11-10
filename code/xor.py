# feedforward neural network trained with backpropagation
# input(2) -> hidden(2) -> output(1)
# trained on the XOR problem
# Paul Gribble, November 2012
# paul [at] gribblelab [dot] org

# ipython --pylab

import time
random.seed(int(time.time()))

# sigmoid activation function
def tansig(x):
	return tanh(x)

# derivative of sigmoid function
# (needed for calculating error gradients using backprop)
def dtansig(x):
	# in case x is a vector, multiply() will do element-wise multiplication
	return 1.0 - (multiply(x,x))

# load up some training examples
#    we will try to get our nnet to learn
#    the XOR mapping
#    http://en.wikipedia.org/wiki/XOR_gate

# numpy matrix of input examples
# 4 training examples, each with 2 inputs
xor_in = matrix([	[0.0, 0.0],
					[0.0, 1.0],
					[1.0, 0.0],
					[1.0, 1.0]	])

# numpy matrix of corresponding outputs
# 4 training examples, each with 1 output
xor_out = matrix([	[0.0],
					[1.0],
					[1.0],
					[0.0]	])

# initialize our nnet : input(2) -> hidden(2) -> output(1)
# initialize network weights to small random values
wgt_hid = rand(2,2)*0.5 - 0.25		# [inp1,inp2] x-> [hid1,hid2]
wgt_out = rand(2,1)*0.5 - 0.25		# [hid1,hid2] x-> [out1]
wgt_out_prev_change = zeros(shape(wgt_out))
wgt_hid_prev_change = zeros(shape(wgt_hid))
maxepochs = 10000
errors = zeros((maxepochs,1))
N = 0.01 # learning rate parameter
M = 0.10 # momentum parameter

# train the sucker!
for i in range(maxepochs):						# iterate over epochs
	net_out = zeros(shape(xor_out))
	for j in range(shape(xor_in)[0]):			# iterate over training examples
		# forward pass
		act_inp = xor_in[j,:]					# select the first training example
		act_hid = tansig( act_inp * wgt_hid )	# hidden unit activations
		act_out = tansig( act_hid * wgt_out )	# output unit activations
		net_out[j,:] = act_out[0,:]

		# error gradients starting from outputs and working backwards
		err_out = (act_out - xor_out[j,:])
		deltas_out = multiply(dtansig(act_out), err_out)
		err_hid = deltas_out * transpose(wgt_out)
		deltas_hid = multiply(dtansig(act_hid), err_hid)

		# update the weights!
		wgt_out_change = -2.0 * transpose(act_hid)*deltas_out
		wgt_out = wgt_out + (N * wgt_out_change) + (M * wgt_out_prev_change)
		wgt_out_prev_change = wgt_out_change
		wgt_hid_change = -2.0 * transpose(act_inp)*deltas_hid
		wgt_hid = wgt_hid + (N * wgt_hid_change) + (M * wgt_hid_prev_change)
		wgt_hid_prev_change = wgt_hid_change

	# compute errors across all targets
	errors[i] = 0.5 * sum(square(net_out - xor_out))
	if ((i % 100)==0):
		print "*** EPOCH %4d/%4d : SSE = %6.5f" % (i,maxepochs,errors[i])
		print net_out

# plot SSE over time
figure()
subplot(2,1,1)
plot(errors)
xlabel('EPOCH')
ylabel('SS_ERROR')
subplot(2,1,2)
plot(log(errors))
xlabel('EPOCH')
ylabel('LOG (SS_ERROR)')














