from scipy.integrate import odeint

# here's our error function
# For a given input x, it returns the error as output
def myFun(x):
	return (x-5.0)**2.0 + 3.0


# here's a list of candidate values to try
values = arange(-10, 20, 0.1)

# initialize an array to store the results
# fill it with zeros for now
out = zeros(size(values))

# loop over the candidate inputs, computing error
# store each one in our out array
for i in arange(0,size(values),1):
	out[i] = myFun(values[i])

# plot the relationship between input and error
plot(values, out)
xlabel('INPUT VALUES')
ylabel('OUTPUT VALUES')

# now let's use an optimizer to find the best input automatically
# it will find the input that minimizes the error function

from scipy import optimize

best_in = optimize.fminbound(myFun, -10.0, 10.0)

###################################

# here is how you might structure your assignment, question #4

# your ode function for baseball simulation
def Baseball(state, t):
	# blha blah blah
	state_d = ...
	return state_d

# your error function that relates initial state [x0,y0] to error
def myErrFun(state0):
	t = ...
	state = odeint(Baseball, state0, t)
	xpos_at_y0 = ....
	error = xpos_at_y0 - 100.0
	return error

state0_guess = [...]
best_state0 = optimize.fmin(myErrFun, state0_guess)


