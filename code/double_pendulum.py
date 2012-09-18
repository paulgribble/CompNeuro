# ipython --pylab

from scipy.integrate import odeint

# here is our state function that implements the
# differential equations defining the dynamics of
# a double-pendulum
def DoublePendulum(x, t):
    damping = 0.0
    g = 9.8
    # inertia matrix
    M = matrix([[3 + 2*cos(x[1]), 1+cos(x[1])], [1+cos(x[1]), 1]])
    # coriolis, centripetal and gravitational forces
    c1 = x[3]*((2*x[2]) + x[3])*sin(x[1]) + 2*g*sin(x[0]) + g*sin(x[0]+x[1])
    c2 = -(x[2]**2)*sin(x[1]) + g*sin(x[0]+x[1])
    # passive dynamics
    cc = [c1-damping*x[2], c2-damping*x[3]]
    u = linalg.solve(M,cc)
    return [x[2], x[3], u[0], u[1]]

# decide on a time range for simulation
# and initial states
t = arange(0, 10, 0.01)
x0 = [pi, pi/2, 0.0, 0.0] # a0, a1, a0d, a1d

# simulate!
x = odeint(DoublePendulum, x0, t)

# plot the states over time
figure()
plot(t,x)
legend(('a0','a1','a0d','a1d'))
xlabel('TIME (sec)')
ylabel('ANGLE (rad)')
draw()

# a utility function to convert from joint angles to hinge positions
# we will use this for our animation
def a2h(a0,a1):
	h0 = [0,0]
	h1 = [np.sin(a0), np.cos(a0)]
	h2 = [np.sin(a0)+np.sin(a0+a1), np.cos(a0)+np.cos(a0+a1)]
	return [h0, h1, h2]

# let's make an animation of the pendulum's motion
def AnimatePendulum(states,t):
	# slice out the two angles and two angular velocities
	a0 = states[:,0]
	a1 = states[:,1]
	a0d = states[:,2]
	a1d = states[:,3]

	# new figure
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(1,2,1)
	ax.plot(t,a0,'b')
	ax.plot(t,a1,'r')
	l, = ax.plot((t[0],t[0]),(-2,5),'k-')
	xlabel('TIME (sec)')
	ylabel('ANGLE (rad)')
	ax = fig.add_subplot(1,2,2)
	(h0,h1,h2) = a2h(a0[0],a1[0]) # convert first time point to hinge positions
	line1, = plot([h0[0], h1[0]],[h0[1], h1[1]],'b')
	line2, = plot([h1[0], h2[0]],[h1[1], h2[1]],'r')
	p0, = plot(h0[0], h0[1], 'k.')
	p1, = plot(h1[0], h1[1], 'b.')
	p2, = plot(h2[0], h2[1], 'r.')
	plt.xlim([-2.2,2.2])
	plt.ylim([-2.2,2.2])
	title1 = title("%3.2fs" % 0.0)
	xlabel('X POSITION (m)')
	ylabel('Y POSITION (m)')
	n = t.size
	for i in arange(0,n,5):
		l.set_xdata((t[i],t[i]))
		(h0,h1,h2) = a2h(a0[i],a1[i])
		line1.set_xdata([h0[0], h1[0]])
		line1.set_ydata([h0[1], h1[1]])
		line2.set_xdata([h1[0], h2[0]])
		line2.set_ydata([h1[1], h2[1]])
		p1.set_xdata(h1[0])
		p1.set_ydata(h1[1])
		p2.set_xdata(h2[0])
		p2.set_ydata(h2[1])
		title1.set_text('time = %3.2fs' %t[i])
		draw()

# Go animation!
AnimatePendulum(x,t)


