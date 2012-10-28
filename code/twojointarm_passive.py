# ipython --pylab

from scipy.integrate import odeint

# forward dynamics equations of our passive two-joint arm
def twojointarm(state,t,aparams):
	"""
	passive two-joint arm in a vertical plane
	X is fwd(+) and back(-)
	Y is up(+) and down(-)
	gravity acts down
	shoulder angle a1 relative to Y vert, +ve counter-clockwise
	elbow angle a2 relative to upper arm, +ve counter-clockwise
	"""
	a1,a2,a1d,a2d = state
	l1,l2 = aparams['l1'], aparams['l2']
	m1,m2 = aparams['m1'], aparams['m2']
	i1,i2 = aparams['i1'], aparams['i2']
	r1,r2 = aparams['r1'], aparams['r2']
	g = 9.81
	M11 = i1 + i2 + (m1*r1*r1) + (m2*((l1*l1) + (r2*r2) + (2*l1*r2*cos(a2))))
	M12 = i2 + (m2*((r2*r2) + (l1*r2*cos(a2))))
	M21 = M12
	M22 = i2 + (m2*r2*r2)
	M = matrix([[M11,M12],[M21,M22]])
	C1 = -(m2*l1*a2d*a2d*r2*sin(a2)) - (2*m2*l1*a1d*a2d*r2*sin(a2))
	C2 = m2*l1*a1d*a1d*r2*sin(a2)
	C = matrix([[C1],[C2]])
	G1 = (g*sin(a1)*((m2*l1)+(m1*r1))) + (g*m2*r2*sin(a1+a2))
	G2 = g*m2*r2*sin(a1+a2)
	G = matrix([[G1],[G2]])
	ACC = inv(M) * (-C-G)
	a1dd,a2dd = ACC[0,0],ACC[1,0]
	return [a1d, a2d, a1dd, a2dd]

# anthropometric parameters of the arm
aparams = {
	'l1' : 0.3384, # metres
	'l2' : 0.4554,
	'r1' : 0.1692,
	'r2' : 0.2277,
	'm1' : 2.10,   # kg
	'm2' : 1.65,
	'i1' : 0.025,  # kg*m*m
	'i2' : 0.075
}

# forward kinematics
def joints_to_hand(A,aparams):
	"""
	Given joint angles A=(a1,a2) and anthropometric params aparams,
	returns hand position H=(hx,hy) and elbow position E=(ex,ey)
	"""
	l1 = aparams['l1']
	l2 = aparams['l2']
	n = shape(A)[0]
	E = zeros((n,2))
	H = zeros((n,2))
	for i in range(n):
		E[i,0] = l1 * cos(A[i,0])
		E[i,1] = l1 * sin(A[i,0])
		H[i,0] = E[i,0] + (l2 * cos(A[i,0]+A[i,1]))
		H[i,1] = E[i,1] + (l2 * sin(A[i,0]+A[i,1]))
	return H,E

def animatearm(state,t,aparams):
	"""
	animate the twojointarm
	"""
	A = state[:,[0,1]]
	A[:,0] = A[:,0] - (pi/2)
	H,E = joints_to_hand(A,aparams)
	l1,l2 = aparams['l1'], aparams['l2']
	figure()
	plot(0,0,'b.')
	p1, = plot(E[0,0],E[0,1],'b.')
	p2, = plot(H[0,0],H[0,1],'b.')
	p3, = plot((0,E[0,0],H[0,0]),(0,E[0,1],H[0,1]),'b-')
	tt = title("%4.2f sec" % 0.00)
	xlim([-l1-l2, l1+l2])
	ylim([-l1-l2, l1+l2])
	skip = 3
	for i in range(len(t)):
		p1.set_xdata((E[i,0]))
		p1.set_ydata((E[i,1]))
		p2.set_xdata((H[i,0]))
		p2.set_ydata((H[i,1]))
		p3.set_xdata((0,E[i,0],H[i,0]))
		p3.set_ydata((0,E[i,1],H[i,1]))
		tt.set_text("%4.2f sec" % (i*0.001))
		draw()
		
state0 = [0*pi/180, 90*pi/180, 0, 0] # initial joint angles and vels
t = arange(10001.)/1000               # 10 seconds at 1000 Hz
state = odeint(twojointarm, state0, t, args=(aparams,))

animatearm(state,t,aparams)
