from sympy import *

m1,m2,r1,r2,i1,i2,l1,l2,a1,a2,t,g = symbols('m1 m2 r1 r2 i1 i2 l1 l2 a1 a2 t g')

# positions
x1 = r1*sin(a1(t))
y1 = -r1*cos(a1(t))
x2 = l1*sin(a1(t)) + r2*sin(a1(t)+a2(t))
y2 = -l1*cos(a1(t)) - r2*cos(a1(t)+a2(t))

# velocities
x1d = diff(x1,t)
y1d = diff(y1,t)
x2d = diff(x2,t)
y2d = diff(y2,t)
a1d = diff(a1(t),t)
a2d = diff(a2(t),t)

# linear kinetic energy
Tlin1 = 0.5 * m1 * ((x1d*x1d) + (y1d*y1d))
Tlin2 = 0.5 * m2 * ((x2d*x2d) + (y2d*y2d))

# rotational kinetic energy
Trot1 = 0.5 * i1 * a1d * a1d
Trot2 = 0.5 * i2 * (a1d+a2d) * (a1d+a2d)

# total kinetic energy
T = Tlin1 + Tlin2 + Trot1 + Trot2

# potential energy
U1 = m1 * g * ( r1*(1-cos(a1(t))) )
U2 = m2 * g * ( l1*(1-cos(a1(t))) + r2*(1-cos(a1(t)-a2(t))) )
U = U1 + U2

# lagrangian L
L = T - U
L = nsimplify(L)

# compute generalized forces (toruqes) Qj
dldq1 = simplify(diff(L,a1(t)))
dldqd1 = simplify(diff(L,diff(a1(t),t)))
ddtdldqd1 = simplify(diff(dldqd1,t))
Q1 = ddtdldqd1 - dldq1

dldq2 = simplify(diff(L,a2(t)))
dldqd2 = simplify(diff(L,diff(a2(t),t)))
ddtdldqd2 = simplify(diff(dldqd2,t))
Q2 = ddtdldqd2 - dldq2

# simplify!
Q1 = simplify(nsimplify(Q1))
Q2 = simplify(nsimplify(Q2))
Q1 = collect(Q1, sin(a1(t)))
Q2 = collect(Q2, sin(a1(t)))
Q1 = collect(Q1, Derivative(Derivative(a1(t),t),t))
Q1 = collect(Q1, Derivative(Derivative(a2(t),t),t))
Q2 = collect(Q2, Derivative(Derivative(a1(t),t),t))
Q2 = collect(Q2, Derivative(Derivative(a2(t),t),t))

pprint(Q1)
pprint(Q2)
