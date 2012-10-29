# assignment 4 solution

def joints_to_hand(a1,a2,l1,l2):
  Ex = l1 * cos(a1)
  Ey = l1 * sin(a1)
  Hx = Ex + (l2 * cos(a1+a2))
  Hy = Ey + (l2 * sin(a1+a2))
  return Ex,Ey,Hx,Hy

def minjerk(H1x,H1y,H2x,H2y,t,n):
  """
  Given hand initial position H1x,H1y, final position H2x,H2y and movement duration t,
  and the total number of desired sampled points n,
  Calculates the hand path H over time T that satisfies minimum-jerk.
  Flash, Tamar, and Neville Hogan. "The coordination of arm
  movements: an experimentally confirmed mathematical model." The
  journal of Neuroscience 5, no. 7 (1985): 1688-1703.
  """
  T = linspace(0,t,n)
  Hx = zeros(n)
  Hy = zeros(n)
  for i in range(n):
    tau = T[i]/t
    Hx[i] = H1x + ((H1x-H2x)*(15*(tau**4) - (6*tau**5) - (10*tau**3)))
    Hy[i] = H1y + ((H1y-H2y)*(15*(tau**4) - (6*tau**5) - (10*tau**3)))
  return T,Hx,Hy


# Question 1
l1 = 0.34
l2 = 0.46
angs = array([30.0,60.0,90.0]) * pi/180
figure(figsize=(5,10))
for i in range(3):
  for j in range(3):
    a1 = angs[i]
    a2 = angs[j]
    subplot(2,1,1)
    plot(a1*180/pi,a2*180/pi,'r+')
    ex,ey,hx,hy = joints_to_hand(a1,a2,l1,l2)
    subplot(2,1,2)
    plot(hx,hy,'r+')
    for k in range(20):
      a1n = a1 + randn()*(sqrt(3)*pi/180)
      a2n = a2 + randn()*(sqrt(3)*pi/180)
      subplot(2,1,1)
      plot(a1n*180/pi,a2n*180/pi,'b.')
      ex,ey,hx,hy = joints_to_hand(a1n,a2n,l1,l2)
      subplot(2,1,2)
      plot(hx,hy,'b.')
subplot(2,1,1)
axis('equal')
xlabel('SHOULDER ANGLE (deg)')
ylabel('ELBOW ANGLE (deg)')
subplot(2,1,2)
axis('equal')
xlabel('HAND POSITION X (m)')
ylabel('HAND POSITION Y (m)')


# Question 2
def hand_to_joints(hx,hy,l1,l2):
  """
  Given hand position H=(hx,hy) and link lengths l1,l2,
  returns joint angles A=(a1,a2)
  """
  a2 = arccos(((hx*hx)+(hy*hy)-(l1*l1)-(l2*l2))/(2.0*l1*l2))
  a1 = arctan(hy/hx) - arctan((l2*sin(a2))/(l1+(l2*cos(a2))))
  if a1 < 0:
    a1 = a1 + pi
  elif a1 > pi:
    a1 = a1 - pi
  return a1,a2


# Question 3
l1 = 0.34
l2 = 0.46
angs = array([30.0,60.0,90.0]) * pi/180
figure(figsize=(5,10))
for i in range(3):
  for j in range(3):
    a1 = angs[i]
    a2 = angs[j]
    subplot(2,1,1)
    plot(a1*180/pi,a2*180/pi,'r+')
    ex,ey,hx,hy = joints_to_hand(a1,a2,l1,l2)
    subplot(2,1,2)
    plot(hx,hy,'r+')
    for k in range(20):
      hxn = hx + randn()*(sqrt(2)/100)
      hyn = hy + randn()*(sqrt(2)/100)
      a1n,a2n = hand_to_joints(hxn,hyn,l1,l2)
      subplot(2,1,1)
      plot(a1n*180/pi,a2n*180/pi,'b.')
      subplot(2,1,2)
      plot(hxn,hyn,'b.')
subplot(2,1,1)
axis('equal')
xlabel('SHOULDER ANGLE (deg)')
ylabel('ELBOW ANGLE (deg)')
title('JOINT SPACE')
subplot(2,1,2)
axis('equal')
xlabel('HAND POSITION X (m)')
ylabel('HAND POSITION Y (m)')
title('HAND SPACE')


# Question 4
l1,l2 = 0.34, 0.46
H1x,H1y = -0.20, -.55
movdist = 0.10
movtime = 0.50
npts = 20
ncirc = 8
angs = linspace(0,360,ncirc+1)*pi/180
angs = angs[0:-1]
figure(figsize=(5,10))
for i in range(ncirc):
  H2x = H1x + movdist*cos(angs[i])
  H2y = H1y + movdist*sin(angs[i])
  T,Hx,Hy = minjerk(H1x,H1y,H2x,H2y,movtime,npts)
  subplot(2,1,2)
  plot(Hx,Hy,'.')
  axis('equal')
  A1 = zeros(npts)
  A2 = zeros(npts)
  for j in range(npts):
    A1[j],A2[j] = hand_to_joints(Hx[j],Hy[j],l1,l2)
    subplot(2,1,1)
    plot(A1*180/pi,A2*180/pi,'.')
    axis('equal')
subplot(2,1,1)
xlabel('SHOULDER ANGLE (deg)')
ylabel('ELBOW ANGLE (deg)')
title('JOINT SPACE')
subplot(2,1,2)
xlabel('HAND POS X (m)')
ylabel('HAND POS Y (m)')
title('HAND SPACE')

# Question 5
l1,l2 = 0.34, 0.46
A1s,A1e = 45*pi/180, 90*pi/180
movdist = 10*pi/180
movtime = 0.50
npts = 20
ncirc = 8
angs = linspace(0,360,ncirc+1)*pi/180
angs = angs[0:-1]
figure(figsize=(5,10))
for i in range(ncirc):
  A2s = A1s + movdist*cos(angs[i])
  A2e = A1e + movdist*sin(angs[i])
  T,As,Ae = minjerk(A1s,A1e,A2s,A2e,movtime,npts)
  subplot(2,1,1)
  plot(As*180/pi,Ae*180/pi,'.')
  axis('equal')
  Hx = zeros(npts)
  Hy = zeros(npts)
  for j in range(npts):
    ex,ey,Hx[j],Hy[j] = joints_to_hand(As[j],Ae[j],l1,l2)
    subplot(2,1,2)
    plot(Hx,Hy,'.')
    axis('equal')
subplot(2,1,1)
xlabel('SHOULDER ANGLE (deg)')
ylabel('ELBOW ANGLE (deg)')
title('JOINT SPACE')
subplot(2,1,2)
xlabel('HAND POS X (m)')
ylabel('HAND POS Y (m)')
title('HAND SPACE')
