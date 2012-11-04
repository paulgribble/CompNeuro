# ipython --pylab

from scipy.integrate import odeint

def onejointmuscle(state,t,a0):
	m = 1.65    # kg
	g = -9.81   # m/s/s
	lz = 0.179  # metres
	I = 0.0779  # kg m**2
	k = -10.0   # Nm/rad
	b =  0.5    # Nms/rad
	a = state[0]
	ad = state[1]
	Tmf = max((k*(a-a0)) - (b*ad),0)
	Tme = min((k*(a-a0)) - (b*ad),0)
	add = (m*g*lz*cos(a) + Tmf + Tme) / I
	return [ad,add]

state0 = [30*pi/180, 0] # 30 deg initial position and 0 deg/s initial velocity
t = linspace(0,5,1001)  # 0 to 5 seconds at 200 Hz
a0 = 30*pi/180          # rest angle for muscles
state = odeint(onejointmuscle, state0, t, args=(a0,))

def animate_arm(state,t):
	l = 0.45
	figure()
	plot(0,0,'k.',markersize=10)
	plot((0,0),(0,.5),'k-',linewidth=2)
	plot((0,.5),(0,0),'k--')
	plot((-0.5,0.5,0.5,-0.5,-0.5),(-0.5,-0.5,0.5,0.5,-0.5),'k-',linewidth=0.5)
	p, = plot((0,l*cos(state[0,0])),(0,l*sin(state[0,0])),'b-')
	tt = title("%4.2f sec" % 0.00)
	xlim([-l-.05,l+.05])
	ylim([-l-.05,l+.05])
	axis('equal')
	step = 3
	tt = title("Click on this plot to continue...")
	ginput(1)
	for i in xrange(1,shape(state)[0]-10,step):
		p.set_xdata((0,l*cos(state[i,0])))
		p.set_ydata((0,l*sin(state[i,0])))
		tt.set_text("%4.2f sec" % (i*0.01))
		draw()

animate_arm(state,t)
