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
figure(figsize=(6,9))
for i in range(ncirc):
  H2x = H1[0] + movdist*cos(angs[i])
  H2y = H1[1] + movdist*sin(angs[i])
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
figure(figsize=(6,9))
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
figure(figsize=(6,9))
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
H1 = [-0.2, 0.4] # hand start position
H2 = [-0.2, 0.6] # hand end position
# get min-jerk hand trajectory
t,H,A,Ad,Add = get_min_jerk_movement(H1,H2,0.500)
# compute torques
Q = inverse_dynamics(A,Ad,Add,aparams)
# initial state of arm for forward simulation
state0 = [A[0,0],A[0,1],Ad[0,0],Ad[0,1]]
# simulate forward dynamics equations of motion with driving torques Q
state = odeint(forward_dynamics, state0, t, args=(aparams, Q, t,))
figure()
plot(t,H)
ylim([-0.25, 0.65])
xlabel('TIME (sec)')
ylabel('HAND POS (m)')
legend(('Hand X','Hand Y'), loc='top left')

# Question 5
figure(figsize=(6,9))
for i in range(25):
	# add noise to initial hand location
	H1n = [H1[0]+(randn()*sqrt(0.001)), H1[1]+(randn()*sqrt(0.001))]
	# convert noisy initial hand loction to joint angles
	A1n = hand_to_joints(matrix(H1n),aparams)
	# initial states for forward simulation
	state0 = [A1n[0,0], A1n[0,1], 0.0, 0.0]
	state = odeint(forward_dynamics, state0, t, args=(aparams, Q, t,))
	Htraj, Etraj = joints_to_hand(state[:,[0,1]], aparams)
	subplot(2,1,1)
	plot(Htraj[:,0],Htraj[:,1],'b-')
	subplot(2,1,2)
	plot(Htraj[0,0],Htraj[0,1],'b.')   # initial hand position
	plot(Htraj[-1,0],Htraj[-1,1],'r.') # final hand position
	draw()
subplot(2,1,1)
axis('equal')
ylabel('HAND POS Y (m)')
subplot(2,1,2)
axis('equal')
xlabel('HAND POS X (m)')
ylabel('HAND POS Y (m)')

# Question 6
figure(figsize=(6,9))
xshifts = linspace(-0.1, 0.1, 11)
xerr = zeros(len(xshifts))
yerr = zeros(len(xshifts))
for i in range(len(xshifts)):
	# shift initial hand x position by specified amount
	H1s = [H1[0]+xshifts[i], H1[1]]
	# convert new initial hand loction to joint angles
	A1s = hand_to_joints(matrix(H1s),aparams)
	# initial states for forward simulation
	state0 = [A1s[0,0], A1s[0,1], 0.0, 0.0]
	# simulate!
	state = odeint(forward_dynamics, state0, t, args=(aparams, Q, t,))
	# get hand trajectory
	Htraj, Etraj = joints_to_hand(state[:,[0,1]], aparams)
	subplot(2,1,1)
	# plot ideal hand trajectory
	plot((Htraj[0,0],H2[0]+xshifts[i]),(Htraj[0,1],H2[1]),'k--')
	# plot actual hand trajectory
	plot(Htraj[:,0],Htraj[:,1],'b-')
	draw()
	# compute endpoint error in x and y
	xerr[i] = Htraj[-1,0] - Htraj[0,0] # endpoint error in x position
	yerr[i] = Htraj[-1,1] - H2[1]   # endpoint error in y position
subplot(2,1,1)
axis('equal')
xlabel('HAND POS X (m)')
ylabel('HAND POS Y (m)')
subplot(2,1,2)
plot((-.12,.12),(0,0),'k--')
plot(xshifts,xerr,'b+-')
plot(xshifts,yerr,'rs-')
legend(('X error','Y error'), loc='upper left')
xlabel('X POSITION SHIFT (m)')
ylabel('ENDPOINT ERROR (m)')
xlim([-.12, .12])

