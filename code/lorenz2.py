from scipy.integrate import odeint

def Lorenz(state,t):
  # unpack the state vector
  x = state[0]
  y = state[1]
  z = state[2]
  
  # these are our constants
  sigma = 10.0
  rho = 28.0
  beta = 8.0/3.0

  # compute state derivatives
  xd = sigma * (y-x)
  yd = (rho-z)*x - y
  zd = x*y - beta*z
  
  # return the state derivatives
  return [xd, yd, zd]

t = arange(0.0, 30, 0.01)

# original initial conditions
state1_0 = [2.0, 3.0, 4.0]
state1 = odeint(Lorenz, state1_0, t)

# rerun with very small change in initial conditions
delta = 0.0001
state2_0 = [2.0+delta, 3.0, 4.0]
state2 = odeint(Lorenz, state2_0, t)

# animation
fig,ax = subplots()
pb, = ax.plot(state1[:,0],state1[:,1],'b-',alpha=0.2)
xlabel('x')
ylabel('y')
p, = ax.plot(state1[0:10,0],state1[0:10,1],'b-')
pp, = ax.plot(state1[10,0],state1[10,1],'b.',markersize=10)
p2, = ax.plot(state2[0:10,0],state2[0:10,1],'r-')
pp2, = ax.plot(state2[10,0],state2[10,1],'r.',markersize=10)
tt = title("%4.2f sec" % 0.00)
# animate
step = 3
for i in xrange(1,shape(state1)[0]-10,step):
  p.set_xdata(state1[10+i:20+i,0])
  p.set_ydata(state1[10+i:20+i,1])
  pp.set_xdata(state1[19+i,0])
  pp.set_ydata(state1[19+i,1])
  p2.set_xdata(state2[10+i:20+i,0])
  p2.set_ydata(state2[10+i:20+i,1])
  pp2.set_xdata(state2[19+i,0])
  pp2.set_ydata(state2[19+i,1])
  tt.set_text("%4.2f sec" % (i*0.01))
  draw()

i = 1939          # the two simulations really diverge here!
s1 = state1[i,:]
s2 = state2[i,:]
d12 = norm(s1-s2) # distance
print ("distance = %f for a %f different in initial condition") % (d12, delta)

