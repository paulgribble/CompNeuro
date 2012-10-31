# Paul Gribble
# paul [at] gribblelab [dot] org
# Oct 31, 2012

import pygame
import numpy
import math

numpy.random.seed(10)

showarm = True

# Define some colors
black    = (   0,   0,   0)
white    = ( 255, 255, 255)
green    = (   0, 255,   0)
red      = ( 255,   0,   0)

targetsize = 30 # radius
ballsize = 5 # radius

# Function for computing distance to target
def dist_to_target(xh,hy,tx,ty):
    return math.sqrt((hx-(tx+targetsize))**2 + (hy-(ty+targetsize))**2)

# Function for displaying text
def printText(txtText, Textfont, Textsize , Textx, Texty, Textcolor):
    # pick a font you have and set its size
    myfont = pygame.font.SysFont(Textfont, Textsize)
    # apply it to text on a label
    label = myfont.render(txtText, 1, Textcolor)
    # put the label object on the screen at point Textx, Texty
    screen.blit(label, (Textx, Texty))

# Function for computing hand pos from joint angles
def joint_to_xy(sx,sy,s,e,l1,l2,ppm):
    exi = sx + (l1*ppm)*math.cos(s)
    eyi = sy - (l1*ppm)*math.sin(s)
    hxi = exi + (l2*ppm)*math.cos(s+e)
    hyi = eyi - (l2*ppm)*math.sin(s+e)
    return exi,eyi,hxi,hyi

# Function to draw arm
def draw_arm(screen,sx,sy,ex,ey,hx,hy,l1,l2):
    pygame.draw.lines(screen,black,False,[(sx,sy),(ex,ey),(hx,hy)],1)

# Function to get a new random target
def getnewtarget(sx,sy,l1,l2,ppm):
    slim = [15*math.pi/180, 90*math.pi/180]
    elim = [45*math.pi/180, 90*math.pi/180]
    stgt = (numpy.random.random() * (slim[1]-slim[0])) + slim[0]
    etgt = (numpy.random.random() * (elim[1]-elim[0])) + elim[0]
    ex,ey,tx,ty = joint_to_xy(sx,sy,stgt,etgt,l1,l2,ppm)
    return tx,ty

# Setup
pygame.init()
   
# Set the width and height of the screen [width,height]
ssize=[800,800]
screen=pygame.display.set_mode(ssize)

pygame.display.set_caption("hit the red targets with the black dot")
  
#Loop until the user clicks the close button.
done=False
  
# Used to manage how fast the screen updates
clock=pygame.time.Clock()
 
# Hide the mouse cursor
pygame.mouse.set_visible(0)
 
# time increment (frames per second)
dt = 30.0
dti = 1.0/dt

# arm geometry
sx = ssize[0] / 2
sy = ssize[1] - (ssize[1]/4)
l1 = 0.34 # metres
l2 = 0.46 # metres
ppm = 500 # pixels per metre
m1 = 2.1  # kg
m2 = 1.65  # kg
i1 = 0.025
i2 = 0.075
# keypress torque [Nm/s * s/frame = Nm/frame]
fsf = 0.10
fse = 0.10
fef = 0.10
fee = 0.10
sfmax = 0.5
semax = 0.5
efmax = 0.5
eemax = 0.5
fleak = 0.90 # "leakage" of muscle force

maxvel = 3.0
maxacc = 30.0

# initial position (radians)
s0 = 45*math.pi/180
e0 = 90*math.pi/180
s=s0
e=e0

# initial vel (rad/s)
sd=0.0
ed=0.0

# initial acc (rad/s/s)
sdd=0.0
edd=0.0

# initial stim values
ksf = 0.0
kse = 0.0
kef = 0.0
kee = 0.0

# initial muscle forces
sf = 0.0
se = 0.0
ef = 0.0
ee = 0.0

# map muscle forces [sf,se,ef,ee] to joint torques [st,et]
M = numpy.matrix([[1,0],[-1,0],[0,1],[0,-1]])

ttotal = 0.0
score = 0
timelimit = 60.0

# target
tx,ty = getnewtarget(sx,sy,l1,l2,ppm)

# -------- Main Program Loop -----------
while done==False:
    if ttotal >= timelimit:
        done=True
    # ALL EVENT PROCESSING SHOULD GO BELOW THIS COMMENT
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done=True # Flag that we are done so we exit this loop
            # User pressed down on a key
         
        if event.type == pygame.KEYDOWN:
            # Figure out if it was an arrow key. If so
            # adjust speed.
            if event.key == pygame.K_d:
                ksf = fsf
            if event.key == pygame.K_f:
                kse = fse
            if event.key == pygame.K_j:
                kef = fef
            if event.key == pygame.K_k:
                kee = fee
            if event.key == pygame.K_SPACE:
                s = s0
                e = e0
                sd = 0
                ed = 0
                sdd = 0
                edd = 0
                sf,se,ef,ee = 0,0,0,0
                score = score-1
                if (score<0):
                    score = 0
            if event.key == pygame.K_TAB:
                showarm = not showarm

        # User let up on a key
        if event.type == pygame.KEYUP:
            # If it is an arrow key, reset vector back to zero
            if event.key == pygame.K_d:
                ksf = 0.0
            if event.key == pygame.K_f:
                kse = 0.0
            if event.key == pygame.K_j:
                kef = 0.0
            if event.key == pygame.K_k:
                kee = 0.0

    # ALL EVENT PROCESSING SHOULD GO ABOVE THIS COMMENT
 
    # ALL GAME LOGIC SHOULD GO BELOW THIS COMMENT

    # convert stim (keypresses) into muscle torque
    sf = min(max((sf + ksf) - (fsf*fleak), 0), sfmax)
    se = min(max((se + kse) - (fse*fleak), 0), semax)
    ef = min(max((ef + kef) - (fef*fleak), 0), efmax)
    ee = min(max((ee + kee) - (fee*fleak), 0), eemax)

    # convert muscle torque into shoulder and elbow torques
    muscles = numpy.matrix([sf,se,ef,ee])
    torque = muscles * M

    # print some values
#    print '-------------------'
#    print ksf, kse, kef, kee
#    print round(sf,2),round(se,2),round(ef,2),round(ee,2)

    # forward dynamics: convert joint torques into joint accelerations
    tmp1 = m2*l1*l2*math.cos(e);
    tmp2 = m2*l2*l2;
    tmp3 = m2*l1*l2*math.sin(e);
    tmp4 = (m2*(l2**2))/4.0;
    A = i1+i2+tmp1+(((m1*l1*l1)+tmp2)/4.0)+(m2*l1*l1);
    B = i2+(tmp2/4.0)+(tmp1/2.0);
    C = tmp3*(ed**2)/2.0;
    D = tmp3*sd*ed;
    E = i2 + (tmp1/2.0)+tmp4;
    F = i2 + tmp4;
    G = (tmp3*(sd**2))/2.0;
    I = numpy.matrix([[A,B],[E,F]])
    T = numpy.matrix([[torque[0,0]],[torque[0,1]]])
    H = numpy.matrix([[-C-D],[G]])
    acc = numpy.linalg.inv(I) * (T-H)
    sdd = acc[0,0]
    edd = acc[1,0]

    # integrate accelerations into vels
    sd = sd + (sdd*dti)
    ed = ed + (edd*dti)

    # integrate vels into positions
    s = s + (sd*dti)
    e = e + (ed*dti)

    ex,ey,hx,hy = joint_to_xy(sx,sy,s,e,l1,l2,ppm)

    # target hit detection
    tardist = dist_to_target(hx,hy,tx,ty)
    if  tardist < targetsize:
        score = score + 1
        tx,ty = getnewtarget(sx,sy,l1,l2,ppm)
 
    # ALL GAME LOGIC SHOULD GO ABOVE THIS COMMENT    
 
    # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
      
    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(white)
    
    if showarm:
        draw_arm(screen,sx,sy,ex,ey,hx,hy,l1,l2)

    pygame.draw.ellipse(screen,red,[tx,ty,targetsize*2,targetsize*2],0)
    pygame.draw.ellipse(screen,black,[hx-ballsize,hy-ballsize,ballsize*2,ballsize*2],0)

    printText("Score:", "MS Comic Sans", 30, 10, 10, red)
    printText(repr(score), "MS Comic Sans", 30, 10, 35, red)

    printText("controls:  d f j k", "MS Comic Sans", 30, ssize[0]/2 - 80, 10, red)
    printText("reset:  SPACEBAR", "MS Comic Sans", 30, ssize[0]/2 - 80, 35, red)
    printText("toggle arm:  TAB", "MS Comic Sans", 30, ssize[0]/2 - 80, 60, red)

    printText("Time:", "MS Comic Sans", 30, ssize[0]-100, 10, red)
    printText(repr(round(timelimit-ttotal,1)), "MS Comic Sans", 30, ssize[0]-100, 35, red)
 
    # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
      
    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
  
    # Limit to dt frames per second
    clock.tick(dt)
    ttotal = ttotal + dti
      
# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit ()
print "your score is: %d" % score

