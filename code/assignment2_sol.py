# assignment 2 solution

from scipy.integrate import odeint

def baseball(state, t):
	k = 5.9e-4
	m = 0.15
	g = 9.81
	x = state[0]
	xd = state[1]
	y = state[2]
	yd = state[3]
	xdd = (-k/m)*xd*sqrt(xd*xd + yd*yd)
	ydd = (-k/m)*yd*sqrt(xd*xd + yd*yd) - g
	return [xd, xdd, yd, ydd]

# set up a time range
t = arange(0, 20, 0.01)

# initial conditions
state0 = [0, 30, 0, 50]

# simulate!
state = odeint(baseball, state0, t)

# find where y < 0
i, = where(y<0)

# find first time y < 0
ifirst = i[0]
time_ground = t[ifirst]
x_ground = state[ifirst,0]

# import optimizer
from scipy import optimize

def myErrFun(vels):
	xd = vels[0]
	yd = vels[1]
	state0 = [0, xd, 0, yd]
	t = arange(0, 20, 0.01)
	state = odeint(baseball, state0, t)
	i, = where(state[:,2]<0)
	ifirst = i[0]
	x_ground = state[ifirst,0]
	err = (x_ground-100.0)**2
	return err

vels_best = optimize.fmin(myErrFun, [25.0, 25.0])

























