from sympy import *

m,r,i,l,a,t,g = symbols('m r i l a t g')
x = r * sin(a(t))
y = -r * cos(a(t))
xd = diff(x,t)
yd = diff(y,t)
Tlin = 0.5 * m * ((xd*xd) + (yd*yd))
Tlin = simplify(Tlin)
ad = diff(a(t),t)
Trot = 0.5 * i * ad * ad
T = Tlin + Trot
T = simplify(T)
U = m * g * r * (r-cos(a(t)))
L = T - U
L = simplify(L)
Q = diff(diff(L,diff(a(t))),t) - diff(L,a(t))
pprint(Q)
