# assignment 6
# due Sunday Dec 2, 11:59pm EST

# 1. explore how (a) number of hidden units, and (b) the cost on the
#    sum of squared weights, affects the network's classification of
#    the input space
#
# 2. (bonus): implement an additional hidden layer and explore how
#    this affects the capability of the network to divide up the input
#    space into classes


# choose a different random seed each time we run the code
import time
myseed = int(time.time())
random.seed(myseed)

# we will need this for conjugate gradient descent
from scipy.optimize import fmin_cg

############################################################
#                   IMPORT TRAINING DATA                   #
############################################################

# inputs: 100 x 2 matrix (100 examples, 2 inputs each)
# outputs: 100 x 1 matrix (100 examples, 1 output each)
# we will want to change outputs to 4:
# outputs: 100 x 4 matrix (100 examples, 4 outputs each)
#          "1" = [1,0,0,0]
#          "2" = [0,1,0,0]
#          "3" = [0,0,1,0]
#          "4" = [0,0,0,1]
#
import pickle
fid = open('traindata.pickle','r')
traindata = pickle.load(fid)
fid.close()
train_in = traindata['inputs']
n_examples = shape(train_in)[0]
out1 = traindata['outputs']
# convert one output value {1,2,3,4} into four binary outputs [o1,o2,o3,o4] {0,1}
train_out = zeros((n_examples,4))
for i in range(n_examples):
	out_i = out1[i,0]
	train_out[i,out_i-1] = 1.0

############################################################
#                    UTILITY FUNCTIONS                     #
############################################################

# The output layer transfer function will be logsig [ 0, +1 ]

def logsig(x):
	""" logsig activation function """
	return 1.0 / (1.0 + exp(-x))

def dlogsig(x):
	""" derivative of logsig function """
	return multiply(x,(1.0 - x))

# The hidden layer transfer function will be tansig [-1, +1 ]

def tansig(x):
	""" tansig activation function """
	return tanh(x)

def dtansig(x):
	""" derivative of tansig function """
	return 1.0 - (multiply(x,x)) # element-wise multiplication

def pack_weights(w_hid, b_hid, w_out, b_out, params):
	""" pack weight matrices into a single vector """
	n_in, n_hid, n_out = params[0], params[1], params[2]
	g_j = hstack((reshape(w_hid,(1,n_in*n_hid)), 
		      reshape(b_hid,(1,n_hid)),
		      reshape(w_out,(1,n_hid*n_out)),
		      reshape(b_out,(1,n_out))))[0]
	g_j = array(g_j[0,:])[0]
	return g_j

def unpack_weights(x, params):
	""" unpack weights from single vector into weight matrices """
	n_in, n_hid, n_out = params[0], params[1], params[2]
	pat_in, pat_out = params[3], params[4]
	n_pat = shape(pat_in)[0]
	i1,i2 = 0,n_in*n_hid
	w_hid = reshape(x[i1:i2], (n_in,n_hid))
	i1,i2 = i2,i2+n_hid
	b_hid = reshape(x[i1:i2],(1,n_hid))
	i1,i2 = i2,i2+(n_hid*n_out)
	w_out = reshape(x[i1:i2], (n_hid,n_out))
	i1,i2 = i2,i2+n_out
	b_out = reshape(x[i1:i2],(1,n_out))
	return w_hid, b_hid, w_out, b_out

def net_forward(x, params, ret_hids=False):
	""" propagate inputs through the network and return outputs """
	w_hid,b_hid,w_out,b_out = unpack_weights(x, params)
	pat_in = params[3]
	hid_act = tansig((pat_in * w_hid) + b_hid)
	out_act = logsig((hid_act * w_out) + b_out)
	if ret_hids:
		return out_act,hid_act
	else:
		return out_act

def f(x,params):
	""" returns the cost (SSE) of a given weight vector """
	t = params[4]
	y = net_forward(x,params)
	sse = sum(square(t-y))
	w_cost = params[5]*sum(square(x))
	cost = sse + w_cost
	print "sse=%7.5f wcost=%7.5f" % (sse,w_cost)
	return cost

def fd(x,params):
	""" returns the gradients (dW/dE) for the weight vector """
	n_in, n_hid, n_out = params[0], params[1], params[2]
	pat_in, pat_out = params[3], params[4]
	w_cost = params[5]
	w_hid,b_hid,w_out,b_out = unpack_weights(x, params)
	act_hid = tansig( (pat_in * w_hid) + b_hid )
	act_out = logsig( (act_hid * w_out) + b_out )
	err_out = act_out - pat_out
	deltas_out = multiply(dlogsig(act_out), err_out)	
	err_hid = deltas_out * transpose(w_out)
	deltas_hid = multiply(dtansig(act_hid), err_hid)
	grad_w_out = transpose(act_hid)*deltas_out
	grad_w_out = grad_w_out + (2*w_cost*grad_w_out)
	grad_b_out = sum(deltas_out,0)
	grad_b_out = grad_b_out + (2*w_cost*grad_b_out)
	grad_w_hid = transpose(pat_in)*deltas_hid
	grad_w_hid = grad_w_hid + (2*w_cost*grad_w_hid)
	grad_b_hid = sum(deltas_hid,0)
	grad_b_hid = grad_b_hid + (2*w_cost*grad_b_hid)
	return pack_weights(grad_w_hid, grad_b_hid, grad_w_out, grad_b_out, params)

############################################################
#                    TRAIN THE SUCKER                      #
############################################################

# network parameters
n_in = shape(train_in)[1]
n_hid = 4
n_out = shape(train_out)[1]
w_cost = 0.01
params = [n_in, n_hid, n_out, train_in, train_out, w_cost]

# initialize weights to small random (uniformly distributed)
# values between -0.10 and +0.10
nw = n_in*n_hid + n_hid + n_hid*n_out + n_out
w0 = random.rand(nw)*0.1 - 0.05

# optimize using conjugate gradient descent
out = fmin_cg(f, w0, fprime=fd, args=(params,),
		 	  full_output=True, retall=True, disp=True,
		 	  gtol=1e-3, maxiter=1000)
# unpack optimizer outputs
wopt,fopt,func_calls,grad_calls,warnflag,allvecs = out

# net performance
netout = net_forward(wopt,params)
pc = array(netout.argmax(1).T) == params[4].argmax(1) # I hate munging numpy matrices/arrays
pc, = where(pc[0,:])								  # hate hate hate
pc = float(len(pc)) / float(shape(params[4])[0])      # more hate
print "percent correct = %6.3f" % (pc)

############################################################
#                     PRETTY PLOTS                         #
############################################################

# test our network on the entire range of inputs
# and visualize the results
#
n_grid = 100
min_grid,max_grid = -10.0, 20.0
g_grid = linspace(min_grid, max_grid, n_grid)
g1,g2 = meshgrid(g_grid, g_grid)
grid_inputs = matrix(hstack((reshape(g1,(n_grid*n_grid,1)),
							 reshape(g2,(n_grid*n_grid,1)))))
params_grid = list(params)
params_grid[3] = grid_inputs
act_grid,hid_grid = net_forward(wopt,params_grid,ret_hids=True)
# choose which neuron has greatest activity
cat_grid = reshape(act_grid.argmax(1),(n_grid,n_grid))
figure()
# plot the network performance
imshow(cat_grid,extent=[min_grid,max_grid,min_grid,max_grid])
# now overlay the training data
i1 = where(traindata['outputs']==1)[0]
i2 = where(traindata['outputs']==2)[0]
i3 = where(traindata['outputs']==3)[0]
i4 = where(traindata['outputs']==4)[0]
plot(traindata['inputs'][i1,0],traindata['inputs'][i1,1],'ys',markeredgecolor='k')
plot(traindata['inputs'][i2,0],traindata['inputs'][i2,1],'rs',markeredgecolor='k')
plot(traindata['inputs'][i3,0],traindata['inputs'][i3,1],'bs',markeredgecolor='k')
plot(traindata['inputs'][i4,0],traindata['inputs'][i4,1],'cs',markeredgecolor='k')
axis([min_grid,max_grid,min_grid,max_grid])
xlabel('INPUT 1')
ylabel('INPUT 2')

# hidden neuron activations for entire range of inputs
#
figure()
ncols = ceil(sqrt(n_hid))
nrows = ceil(float(n_hid)/float(ncols))
w_hid, b_hid, w_out, b_out = unpack_weights(wopt, params)
for i in range(n_hid):
	cgi = reshape(hid_grid[:,i], (n_grid,n_grid))
	subplot(nrows,ncols,i+1)
	imshow(cgi, extent=[min_grid,max_grid,min_grid,max_grid])
	axis([min_grid,max_grid,min_grid,max_grid])
	axis('off')
	title('HID_%d' % i)

# output neuron activations for entire range of inputs
#
figure(figsize=(16,4))
for i in range(4):
	cgi = reshape(act_grid[:,i], (n_grid,n_grid))
	subplot(1,4,i+1)
	imshow(cgi, extent=[min_grid,max_grid,min_grid,max_grid])
	plot(traindata['inputs'][i1,0],traindata['inputs'][i1,1],'ys',markeredgecolor='k')
	plot(traindata['inputs'][i2,0],traindata['inputs'][i2,1],'rs',markeredgecolor='k')
	plot(traindata['inputs'][i3,0],traindata['inputs'][i3,1],'bs',markeredgecolor='k')
	plot(traindata['inputs'][i4,0],traindata['inputs'][i4,1],'cs',markeredgecolor='k')
	axis([min_grid,max_grid,min_grid,max_grid])
	xlabel('INPUT 1')
	ylabel('INPUT 2')
	title('OUT_%d' % i)









