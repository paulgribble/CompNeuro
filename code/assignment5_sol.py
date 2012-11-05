# ipython --pylab

# load in the functions and parameters from twojointarm.py
%load twojointarm.py

# Question 1
H1 = [-0.20, -.55]
movdist = 0.10
movtime = 0.50
npts = 20
ncirc = 8
angs = linspace(0,360,ncirc+1)*pi/180
angs = angs[0:-1]
figure()
for i in range(ncirc):
  H2x = H1x + movdist*cos(angs[i])
  H2y = H1y + movdist*sin(angs[i])
  H2 = [H2x,H2y]
  t,H,A,Ad,Add = get_min_jerk_movement(H1,H2,mt)
  Q = inverse_dynamics(A,Ad,Add,aparams)
  subplot(2,1,1)
  plot(H[:,0],H[:,1],'b.-')
  subplot(2,1,2)
  plot(Q[:,0],Q[:,1],'r.-')
  draw()
subplot(2,1,1)
axis('equal')
xlabel('HAND POS X (m)')
ylabel('HAND POS Y (m)')
title('HAND SPACE')
subplot(2,1,2)
axis('equal')
xlabel('SHOULDER TORQUE (Nm)')
ylabel('ELBOW TORQUE (Nm)')
title('JOINT TORQUE SPACE')

# Question 2
A1 = [45*pi/180, 110*pi/180] # starting angles
A2 = [45*pi/180,  70*pi/180] # ending angles
movtime = 0.500              # 500 ms movement time
npts = 100					 # use 100 pts for our desired trajectory
# get a min-jerk (joint-space) trajectory
t,A,Ad,Add = minjerk(A1,A2,movtime,100)
# compute required joint torques
Q = inverse_dynamics(A,Ad,Add,aparams)
# plot the junk
figure()
subplot(2,1,1)
plot(t,A*180/pi)
ylim([40,120])
legend(('shoulder','elbow'))
ylabel('JOINT ANGLE (deg)')
title('JOINT ANGLES')
subplot(2,1,2)
plot(t,Q)
xlabel('TIME (sec)')
ylabel('JOINT TORQUES (Nm)')
legend(('shoulder','elbow'), loc='upper left')

# Question 3
A1 = [30*pi/180, 90*pi/180] # starting angles
A2 = [60*pi/180, 90*pi/180] # ending angles
movtime = 0.500              # 500 ms movement time
npts = 100					 # use 100 pts for our desired trajectory
# get a min-jerk (joint-space) trajectory
t,A,Ad,Add = minjerk(A1,A2,movtime,100)
# compute required joint torques
Q = inverse_dynamics(A,Ad,Add,aparams)
# plot the junk
figure()
subplot(2,1,1)
plot(t,A*180/pi)
ylim([20,100])
legend(('shoulder','elbow'), loc='bottom right')
ylabel('JOINT ANGLE (deg)')
title('JOINT ANGLES')
subplot(2,1,2)
plot(t,Q)
xlabel('TIME (sec)')
ylabel('JOINT TORQUES (Nm)')
legend(('shoulder','elbow'), loc='upper right')

# Question 4
H1 = [-0.2, 0.4]
H2 = [-0.2, 0.6]
t,H,A,Ad,Add = get_min_jerk_movement(H1,H2,0.500)
Q = inverse_dynamics(A,Ad,Add,aparams)
state0 = [A[0,0],A[0,1],Ad[0,0],Ad[0,1]]
state = odeint(forward_dynamics, state0, t, args=(aparams, Q, t,))
figure()
plot(t,H)
ylim([-0.25, 0.65])
xlabel('TIME (sec)')
ylabel('HAND POS (m)')
legend(('Hand X','Hand Y'), loc='top left')

# Question 5



