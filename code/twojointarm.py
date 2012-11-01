# ipython --pylab

# two joint arm in a horizontal plane, no gravity

# compute a min-jerk trajectory
def minjerk(H1,H2,t,n):
	"""
	Given hand initial position H1=(x1,y1), final position H2=(x2,y2) and movement duration t,
	and the total number of desired sampled points n,
	Calculates the hand path H over time T that satisfies minimum-jerk.
	Also returns derivatives Hd and Hdd
	
	Flash, Tamar, and Neville Hogan. "The coordination of arm
        movements: an experimentally confirmed mathematical model." The
        journal of Neuroscience 5, no. 7 (1985): 1688-1703.
	"""
	T = linspace(0,t,n)
	H = zeros((n,2))
	Hd = zeros((n,2))
	Hdd = zeros((n,2))
	for i in range(n):
		tau = T[i]/t
		H[i,0] = H1[0] + ((H1[0]-H2[0])*(15*(tau**4) - (6*tau**5) - (10*tau**3)))
		H[i,1] = H1[1] + ((H1[1]-H2[1])*(15*(tau**4) - (6*tau**5) - (10*tau**3)))
		Hd[i,0] = (H1[0] - H2[0])*(-30*T[i]**4/t**5 + 60*T[i]**3/t**4 - 30*T[i]**2/t**3)
		Hd[i,1] = (H1[1] - H2[1])*(-30*T[i]**4/t**5 + 60*T[i]**3/t**4 - 30*T[i]**2/t**3)
		Hdd[i,0] = (H1[0] - H2[0])*(-120*T[i]**3/t**5 + 180*T[i]**2/t**4 - 60*T[i]/t**3)
		Hdd[i,1] = (H1[1] - H2[1])*(-120*T[i]**3/t**5 + 180*T[i]**2/t**4 - 60*T[i]/t**3)
	return T,H,Hd,Hdd

# forward kinematics
def joints_to_hand(A,aparams):
	"""
	Given joint angles A=(a1,a2) and anthropometric params aparams,
	returns hand position H=(hx,hy) and elbow position E=(ex,ey)
	Note: A must be type matrix
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

# inverse kinematics
def hand_to_joints(H,aparams):
	"""
	Given hand position H=(hx,hy) and anthropometric params aparams,
	returns joint angles A=(a1,a2)
	Note: H must be type matrix
	"""
	l1 = aparams['l1']
	l2 = aparams['l2']
	n = shape(H)[0]
	A = zeros((n,2))
	for i in range(n):
		A[i,1] = arccos(((H[i,0]*H[i,0])+(H[i,1]*H[i,1])-(l1*l1)-(l2*l2))/(2.0*l1*l2))
		A[i,0] = arctan(H[i,1]/H[i,0]) - arctan((l2*sin(A[i,1]))/(l1+(l2*cos(A[i,1]))))
		if A[i,0] < 0:
			A[i,0] = A[i,0] + pi
		elif A[i,0] > pi:
			A[i,0] = A[i,0] - pi
	return A

# jacobian matrix J(q) = dx/da
def jacobian(A,aparams):
	"""
	Given joint angles A=(a1,a2)
	returns the Jacobian matrix J(q) = dx/dA
	"""
	l1 = aparams['l1']
	l2 = aparams['l2']
	dx1dA1 = -l1*sin(A[0]) - l2*sin(A[0]+A[1])
	dx1dA2 = -l2*sin(A[0]+A[1])
	dx2dA1 = l1*cos(A[0]) + l2*cos(A[0]+A[1])
	dx2dA2 = l2*cos(A[0]+A[1])
	J = matrix([[dx1dA1,dx1dA2],[dx2dA1,dx2dA2]])
	return J

# jacobian matrix Jd(q)
def jacobiand(A,Ad,aparams):
	"""
	Given joint angles A=(a1,a2) and velocities Ad=(a1d,a2d)
	returns the time derivative of the Jacobian matrix d/dt (J)
	"""
	l1 = aparams['l1']
	l2 = aparams['l2']
	Jd11 = -l1*cos(A[0])*Ad[0] - l2*(Ad[0] + Ad[1])*cos(A[0] + A[1])
	Jd12 = -l2*(Ad[0] + Ad[1])*cos(A[0] + A[1])
	Jd21 = -l1*sin(A[0])*Ad[0] - l2*(Ad[0] + Ad[1])*sin(A[0] + A[1])
	Jd22 = -l2*(Ad[0] + Ad[1])*sin(A[0] + A[1])
	Jd = matrix([[Jd11, Jd12],[Jd21, Jd22]])
	return Jd

# utility function for interpolating torque inputs
def getTorque(TorquesIN, TorquesTIME, ti):
	"""
	Given a desired torque command (TorquesIN) defined over a time vector (TorquesTIME),
	returns an interpolated torque command at an intermediate time point ti
	Note: TorquesIN and TorquesTIME must be type matrix
	"""
	t1 = interp(ti, TorquesTIME, TorquesIN[:,0])
	t2 = interp(ti, TorquesTIME, TorquesIN[:,1])
	return matrix([[t1],[t2]])

# utility function for computing some limb dynamics terms
def compute_dynamics_terms(A,Ad,aparams):
	"""
	Given a desired set of joint angles A=(a1,a2) and joint velocities Ad=(a1d,a2d),
	returns M and C matrices associated with inertial and centrifugal/coriolis terms
	"""
	a1,a2,a1d,a2d = A[0],A[1],Ad[0],Ad[1]
	l1,l2 = aparams['l1'], aparams['l2']
	m1,m2 = aparams['m1'], aparams['m2']
	i1,i2 = aparams['i1'], aparams['i2']
	r1,r2 = aparams['r1'], aparams['r2']
	M11 = i1 + i2 + (m1*r1*r1) + (m2*((l1*l1) + (r2*r2) + (2*l1*r2*cos(a2))))
	M12 = i2 + (m2*((r2*r2) + (l1*r2*cos(a2))))
	M21 = M12
	M22 = i2 + (m2*r2*r2)
	M = matrix([[M11,M12],[M21,M22]])
	C1 = -(m2*l1*a2d*a2d*r2*sin(a2)) - (2*m2*l1*a1d*a2d*r2*sin(a2))
	C2 = m2*l1*a1d*a1d*r2*sin(a2)
	C = matrix([[C1],[C2]])
	return M,C

# inverse dynamics
def inverse_dynamics(A,Ad,Add,aparams):
	"""
	inverse dynamics of a two-link planar arm
	Given joint angles A=(a1,a2), velocities Ad=(a1d,a2d) and accelerations Add=(a1dd,a2dd),
	returns joint torques Q required to generate that movement
	Note: A, Ad and Add must be type matrix
	"""
	n = shape(A)[0]
	T = zeros((n,2))
	for i in range(n):
		M,C = compute_dynamics_terms(A[i,:],Ad[i,:],aparams)
		ACC = matrix([[Add[i,0]],[Add[i,1]]])
		Qi = M*ACC + C
		T[i,0],T[i,1] = Qi[0,0],Qi[1,0]
	return T

# forward dynamics
def forward_dynamics(state, t, aparams, TorquesIN, TorquesTIME):
	"""
	forward dynamics of a two-link planar arm
	note: TorquesIN and TorquesTIME must be type matrix
	"""
	a1, a2, a1d, a2d = state   # unpack the four state variables
	Q = getTorque(TorquesIN, TorquesTIME, t)
	M,C = compute_dynamics_terms(state[0:2],state[2:4],aparams)
	# Q = M*ACC + C
	ACC = inv(M) * (Q-C)
	return [a1d, a2d, ACC[0,0], ACC[1,0]]

# Utility function to return hand+joint kinematics for
# a min-jerk trajectory between H1 and H2 in movtime with
# time padding padtime at beginning and end of movement
def get_min_jerk_movement(H1,H2,movtime,padtime=0.2):
	# create a desired min-jerk hand trajectory
	t,H,Hd,Hdd = minjerk(H1,H2,movtime,100)
	# pad it with some hold time on each end
	t = append(append(0.0, t+padtime), t[-1]+padtime+padtime)
	H = vstack((H[0,:],H,H[-1,:]))
	Hd = vstack((Hd[0,:],Hd,Hd[-1,:]))
	Hdd = vstack((Hdd[0,:],Hdd,Hdd[-1,:]))
	# interpolate to get equal spacing over time
	ti = linspace(t[0],t[-1],100)
	hxi = interp(ti, t, H[:,0])
	hyi = interp(ti, t, H[:,1])
	H = zeros((len(ti),2))
	H[:,0],H[:,1] = hxi,hyi
	hxdi = interp(ti, t, Hd[:,0])
	hydi = interp(ti, t, Hd[:,1])
	Hd = zeros((len(ti),2))
	Hd[:,0],Hd[:,1] = hxdi,hydi
	hxddi = interp(ti, t, Hdd[:,0])
	hyddi = interp(ti, t, Hdd[:,1])
	Hdd = zeros((len(ti),2))
	Hdd[:,0],Hdd[:,1] = hxddi,hyddi
	t = ti
	A = zeros((len(t),2))
	Ad = zeros((len(t),2))
	Add = zeros((len(t),2))
	# use inverse kinematics to compute desired joint angles
	A = hand_to_joints(H,aparams)
	# use jacobian to transform hand vels & accels to joint vels & accels
	for i in range(len(t)):
		J = jacobian(A[i,:],aparams)
		Ad[i,:] = transpose(inv(J) * matrix([[Hd[i,0]],[Hd[i,1]]]))
		Jd = jacobiand(A[i,:],Ad[i,:],aparams)
		b = matrix([[Hdd[i,0]],[Hdd[i,1]]]) - Jd*matrix([[Ad[i,0]],[Ad[i,1]]])
		Add[i,:] = transpose(inv(J) * b)
	return t,H,A,Ad,Add

# utility function to plot a trajectory
def plot_trajectory(t,H,A):
	"""
	Note: H and A must be of type matrix
	"""
	hx,hy = H[:,0],H[:,1]
	a1,a2 = A[:,0],A[:,1]
	figure()
	subplot(2,2,1)
	plot(t,hx,t,hy)
	ylim(min(min(hx),min(hy))-0.03, max(max(hx),max(hy))+0.03)
	xlabel('TIME (sec)')
	ylabel('HAND POS (m)')
	legend(('Hx','Hy'))
	subplot(2,2,2)
	plot(hx,hy,'.')
	axis('equal')
	plot(hx[0],hy[0],'go',markersize=8)
	plot(hx[-1],hy[-1],'ro',markersize=8)
	xlabel('HAND X POS (m)')
	ylabel('HAND Y POS (m)')
	subplot(2,2,3)
	plot(t,a1*180/pi,t,a2*180/pi)
	ylim(min(min(a1),min(a1))*180/pi - 5, max(max(a2),max(a2))*180/pi + 5)
	xlabel('TIME (sec)')
	ylabel('JOINT ANGLE (deg)')
	legend(('a1','a2'))
	subplot(2,2,4)
	plot(a1*180/pi,a2*180/pi,'.')
	plot(a1[0]*180/pi,a2[0]*180/pi,'go',markersize=8)
	plot(a1[-1]*180/pi,a2[-1]*180/pi,'ro',markersize=8)
	axis('equal')
	xlabel('SHOULDER ANGLE (deg)')
	ylabel('ELBOW ANGLE (deg)')

def animatearm(state,t,aparams,step=3,crumbs=0):
	"""
	animate the twojointarm
	"""
	A = state[:,[0,1]]
	A[:,0] = A[:,0]
	H,E = joints_to_hand(A,aparams)
	l1,l2 = aparams['l1'], aparams['l2']
	figure()
	plot(0,0,'b.')
	p1, = plot(E[0,0],E[0,1],'b.')
	p2, = plot(H[0,0],H[0,1],'b.')
	p3, = plot((0,E[0,0],H[0,0]),(0,E[0,1],H[0,1]),'b-')
	xlim([-l1-l2, l1+l2])
	ylim([-l1-l2, l1+l2])
	dt = t[1]-t[0]
	tt = title("Click on this plot to continue...")
	ginput(1)
	for i in xrange(0,shape(state)[0]-step,step):
		p1.set_xdata((E[i,0]))
		p1.set_ydata((E[i,1]))
		p2.set_xdata((H[i,0]))
		p2.set_ydata((H[i,1]))
		p3.set_xdata((0,E[i,0],H[i,0]))
		p3.set_ydata((0,E[i,1],H[i,1]))
		if crumbs==1:
			plot(H[i,0],H[i,1],'b.')
		tt.set_text("%4.2f sec" % (i*dt))
		draw()


##############################################################################
#############################  THE FUN PART  #################################
##############################################################################

# anthropometric parameters of the arm
aparams = {
	'l1' : 0.3384, # metres
	'l2' : 0.4554,
	'r1' : 0.1692,
	'r2' : 0.2277,
	'm1' : 2.10,   # kg
	'm2' : 1.65,
	'i1' : 0.025,  # kg*m*m
	'i2' : 0.075,
}

# Get a desired trajectory between two arm positions defined by
# a min-jerk trajectory in Hand-space

H1 = [-0.2, 0.4]  # hand initial position
H2 = [-0.2, 0.6]  # hand final target
mt = 0.500        # 500 milliseconds movement time

# get min-jerk desired kinematic trajectory

t,H,A,Ad,Add = get_min_jerk_movement(H1,H2,mt)
plot_trajectory(t,H,A)

# now compute required joint torques using inverse dynamics equations of motion

TorquesIN = inverse_dynamics(A,Ad,Add,aparams)
figure()
plot(t,TorquesIN)
legend(('torque1','torque2'))

# now do a forward simulation using forward dynamics equations of motion
# just to demonstrate that indeed the TorquesIN do in fact generate
# the desired arm movement

from scipy.integrate import odeint
from scipy.interpolate import interp1d
state0 = [A[0,0], A[0,1], Ad[0,0], Ad[0,1]]
tt = linspace(t[0],t[-1],100)
state = odeint(forward_dynamics, state0, tt, args=(aparams, TorquesIN, t,))

# run through forward kinematics equations to get hand trajectory and plot

Hsim,Esim = joints_to_hand(state,aparams)
plot_trajectory(tt,Hsim,state[:,[0,1]])

animatearm(state,tt,aparams)

