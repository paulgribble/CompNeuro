from scipy.integrate import odeint

def LotkaVolterra(state,t):
  x = state[0]
  y = state[1]
  alpha = 0.1
  beta =  0.1
  sigma = 0.1
  gamma = 0.1
  xd = x*(alpha - beta*y)
  yd = -y*(gamma - sigma*x)
  return [xd,yd]

t = arange(0,500,1)
state0 = [0.5,0.5]
state = odeint(LotkaVolterra,state0,t)
figure()
plot(t,state)
ylim([0,8])
xlabel('Time')
ylabel('Population Size')
legend(('x (prey)','y (predator)'))
title('Lotka-Volterra equations')

# animation in state-space
figure()
pb, = plot(state[:,0],state[:,1],'b-',alpha=0.2)
xlabel('x (prey population size)')
ylabel('y (predator population size)')
p, = plot(state[0:10,0],state[0:10,1],'b-')
pp, = plot(state[10,0],state[10,1],'b.',markersize=10)
tt = title("%4.2f sec" % 0.00)

# animate
step=2
for i in xrange(1,shape(state)[0]-10,step):
  p.set_xdata(state[10+i:20+i,0])
  p.set_ydata(state[10+i:20+i,1])
  pp.set_xdata(state[19+i,0])
  pp.set_ydata(state[19+i,1])
  tt.set_text("%d steps" % (i))
  draw()

