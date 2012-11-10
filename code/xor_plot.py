# feedforward neural network trained with backpropagation
# input(2+bias) -> hidden(2+bias) -> output(1)
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
xor_in = matrix([[0.0, 0.0],
		 [0.0, 1.0],
		 [1.0, 0.0],
		 [1.0, 1.0]	])

# numpy matrix of corresponding outputs
# 4 training examples, each with 1 output
xor_out = matrix([[0.0],
		  [1.0],
		  [1.0],
		  [0.0]	])

# out nnet: input(2+bias) -> hidden(2+bias) -> output(1)
# initialize weights and biases to small random values
sigw = 0.5
w_hid = rand(2,2)*sigw		# [inp1,inp2] x-> [hid1,hid2]
b_hid = rand(1,2)*sigw		# 1.0 -> [b_hid1,b_hid2]
w_out = rand(2,1)*sigw		# [hid1,hid2] x-> [out1]
b_out = rand(1,1)*sigw		# 1.0 -> [b_out1]
w_out_prev_change = zeros(shape(w_out))
b_out_prev_change = zeros(shape(b_out))
w_hid_prev_change = zeros(shape(w_hid))
b_hid_prev_change = zeros(shape(b_hid))

maxepochs = 1000
errors = zeros((maxepochs,1))
N = 0.01 # learning rate parameter
M = 0.10 # momentum parameter

# we are going to plot the network's performance over the course of learning
# inputs will be a regular grid of [inp1,inp2] points
n_grid = 20
g_grid = linspace(-1.0, 2.0, n_grid)
g1,g2 = meshgrid(g_grid, g_grid)
figure()

# train the sucker!
for i in range(maxepochs):
	net_out = zeros(shape(xor_out))
	for j in range(shape(xor_in)[0]): # for each training example
		# forward pass
		act_inp = xor_in[j,:]
		act_hid = tansig( (act_inp * w_hid) + b_hid )
		act_out = tansig( (act_hid * w_out) + b_out )
		net_out[j,:] = act_out[0,:]

		# error gradients starting at outputs and working backwards
		err_out = (act_out - xor_out[j,:])
		deltas_out = multiply(dtansig(act_out), err_out)
		err_hid = deltas_out * transpose(w_out)
		deltas_hid = multiply(dtansig(act_hid), err_hid)

		# update the weights and bias units
		w_out_change = -2.0 * transpose(act_hid)*deltas_out
		w_out = w_out + (N * w_out_change) + (M * w_out_prev_change)
		w_out_prev_change = w_out_change
		b_out_change = -2.0 * deltas_out
		b_out = b_out + (N * b_out_change) + (M * b_out_prev_change)
		b_out_prev_change = b_out_change

		w_hid_change = -2.0 * transpose(act_inp)*deltas_hid
		w_hid = w_hid + (N * w_hid_change) + (M * w_hid_prev_change)
		w_hid_prev_change = w_hid_change
		b_hid_change = -2.0 * deltas_hid
		b_hid = b_out + (N * b_hid_change) + (M * b_hid_prev_change)
		b_hid_prev_change = b_hid_change

	# compute errors across all targets
	errors[i] = 0.5 * sum(square(net_out - xor_out))
	if ((i % 2)==0):
		print "*** EPOCH %4d/%4d : SSE = %6.5f" % (i,maxepochs,errors[i])
		print net_out
		# now do our plotting
		net_perf = zeros(shape(g1))
		for i1 in range(n_grid):
			for i2 in range(n_grid):
				act_inp = matrix([g1[i1,i2],g2[i1,i2]])
				o_grid = tansig( (tansig( (act_inp * w_hid) + b_hid ) * w_out) + b_out )
				o_grid = int(o_grid >= 0.50) # hardlim
				net_perf[i1,i2] = o_grid
		cla()
		imshow(net_perf, extent=[-1,2,-1,2])
		plot((0,0,1,1),(0,1,0,1),'ws',markersize=10)
		axis([-1, 2, -1, 2])
		draw()


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














