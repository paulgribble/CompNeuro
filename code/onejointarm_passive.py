from scipy.integrate import odeint

def onejointarm(state,t):
	theta = state[0]      # joint angle (rad)
	theta_dot = state[1]  # joint velocity (rad/s)
	l = 0.50              # link length (m)
	g = 9.81              # gravitational constant (m/s/s)
	theta_ddot = -g*sin(theta) / l
	return [theta_dot, theta_ddot]

t = linspace(0.0,10.0,1001)    # 10 seconds sampled at 1000 Hz
state0 = [90.0*pi/180.0, 0.0] # 90 deg initial angle, 0 deg/sec initial velocity
state = odeint(onejointarm, state0, t)

figure()
plot(t,state*180/pi)
legend(('theta','thetadot'))
xlabel('TIME (sec)')
ylabel('THETA (deg) & THETA_DOT (deg/sec)')

def animate_arm(state,t):
	l = 0.5
	figure(figsize=(12,6))
	plot(0,0,'r.')
	p, = plot((0,l*sin(state[0,0])),(0,-l*cos(state[0,0])),'b-')
	tt = title("%4.2f sec" % 0.00)
	xlim([-l-.05,l+.05])
	ylim([-l,.10])
	step = 3
	for i in xrange(1,shape(state)[0]-10,step):
		p.set_xdata((0,l*sin(state[i,0])))
		p.set_ydata((0,-l*cos(state[i,0])))
		tt.set_text("%4.2f sec" % (i*0.01))
		draw()

animate_arm(state,t)
