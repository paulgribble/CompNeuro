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

def jacobian(A,aparams):
   """
   Given joint angles A=(a1,a2)
   returns the Jacobian matrix J(q) = dH/dA
   """
   l1 = aparams['l1']
   l2 = aparams['l2']
   dHxdA1 = -l1*sin(A[0]) - l2*sin(A[0]+A[1])
   dHxdA2 = -l2*sin(A[0]+A[1])
   dHydA1 = l1*cos(A[0]) + l2*cos(A[0]+A[1])
   dHydA2 = l2*cos(A[0]+A[1])
   J = matrix([[dHxdA1,dHxdA2],[dHydA1,dHydA2]])
   return J

aparams = {'l1' : 0.3384, 'l2' : 0.4554}

npts = 10
angs = linspace(10.0,120.0,npts) *pi/180.0
A1,A2 = meshgrid(angs,angs)

figure()
# visualize +ve shoulder angle velocity in joint-space and in hand-space
dA1 = ones((npts,npts)) * (5.0*pi/180.0)
dA2 = ones((npts,npts)) * (0.0*pi/180.0)
dHx = zeros((npts,npts))
dHy = zeros((npts,npts))
Hx = zeros((npts,npts))
Hy = zeros((npts,npts))
for i in range(npts):
	for j in range(npts):
		J = jacobian(array([A1[i,j],A2[i,j]]),aparams)
		dA = matrix([[dA1[i,j]],[dA2[i,j]]])
		dH = J * dA
		dHx[i,j], dHy[i,j] = dH[0,0], dH[1,0]
		aij = matrix([[A1[i,j],A2[i,j]]])
		h,e = joints_to_hand(aij,aparams)
		Hx[i,j], Hy[i,j] = h[0,0], h[0,1]
subplot(2,2,1)
quiver(A1*180/pi,A2*180/pi,dA1*180/pi,dA2*180/pi,color='b')
xlabel('SHOULDER ANGLE (deg)')
ylabel('ELBOW ANGLE (deg)')
title('PURE SHOULDER VELOCITY')
subplot(2,2,3)
quiver(Hx,Hy,dHx,dHy,color='r')
xlabel('HAND POS X (m)')
ylabel('HAND POS Y (m)')
# visualize +ve elbow angle velocity in joint-space and in hand-space
dA1 = ones((npts,npts)) * (0.0*pi/180.0)
dA2 = ones((npts,npts)) * (5.0*pi/180.0)
dHx = zeros((npts,npts))
dHy = zeros((npts,npts))
Hx = zeros((npts,npts))
Hy = zeros((npts,npts))
for i in range(npts):
	for j in range(npts):
		J = jacobian(array([A1[i,j],A2[i,j]]),aparams)
		dA = matrix([[dA1[i,j]],[dA2[i,j]]])
		dH = J * dA
		dHx[i,j], dHy[i,j] = dH[0,0], dH[1,0]
		aij = matrix([[A1[i,j],A2[i,j]]])
		h,e = joints_to_hand(aij,aparams)
		Hx[i,j], Hy[i,j] = h[0,0], h[0,1]
subplot(2,2,2)
quiver(A1*180/pi,A2*180/pi,dA1*180/pi,dA2*180/pi,color='b')
xlabel('SHOULDER ANGLE (deg)')
ylabel('ELBOW ANGLE (deg)')
title('PURE ELBOW VELOCITY')
subplot(2,2,4)
quiver(Hx,Hy,dHx,dHy,color='r')
xlabel('HAND POS X (m)')
ylabel('HAND POS Y (m)')

