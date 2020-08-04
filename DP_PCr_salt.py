from sympy import Matrix, symbols, sin, cos, acos
from sympy.physics.mechanics import dynamicsymbols

###
# State and parameter definitions

# define the state
# q = [ \theta_1, \theta_2]

th1, th2 = dynamicsymbols('theta_1, theta_2')
dth1, dth2 = dynamicsymbols('theta_1, theta_2', 1)
q = Matrix([[th1, th2]])
dq = Matrix([[dth1, dth2]])
x = Matrix.hstack(q, dq)

# paramters
l1, l2 = symbols('l_1, l_2', positive=True)
m1, m2 = symbols('m_1, m_2', positive=True)


####
# mass matrix and Christoffel symbols
# Murray et. al 1994. Chpt 4 Section 2.3

a = 1/3*m1*l1**2 + 1/3*m2*l2**2 + m1*(l1/2)**2 + m2*(l1**2+(l2/2)**2)
b = m2*l1*l2/2
d = 1/3*m2*l2**2 + m2*(l2/2)**2

m11 = a+2*b*cos(th2)
m12 = d+b*cos(th2)
m22 = d

M = Matrix([[m11, m12], [m12, m22]])
Minv = M.inv()

c11 = -b*sin(th2)*dth2
c12 = -b*sin(th2)*(dth1+dth2)
c21 = b*sin(th2)*dth1
c22 = 0

Cor = Matrix([[c11, c12], [c21, c22]])

###
# parameter values
l1_val = 1/2
l2_val = 2/7
m1_val = m2_val = 1
corest = .3  # coefficient of restitution

param_vals = {l1:l1_val, l2:l2_val, m1:m1_val, m2:m2_val}

Minv = Minv.subs(param_vals)
Cor = Cor.subs(param_vals)

# values of the state at the simultaneous unilateral constraint activation
th1_act_val = 0
th2_act_val = acos(-d/b).subs(param_vals)
rho = Matrix([th1_act_val, th2_act_val])
rho_subs = {th1:rho[0], th2:rho[1]}
drho = Matrix([-1, 1])

####
# Define the system

# Define the guards

a1 = th1 - th1_act_val
a2 = -th2 + th2_act_val

Da1 = a1.diff(q)
Da2 = a2.diff(q)

#check orthogonality of constraints at simulatenous activation
ortho_check = Da1@Minv@Da2.T
assert ortho_check.subs(param_vals).subs({th1:rho[0], th2:rho[1]})[0, 0] == 0

#define velocity reset

def Delta(DaJ, gamma=corest, evaluate=False):
    '''
    \Delta(q, \dot{q}) = Id - (1+gamma(q,dq)) M^-1(q) Da(q)^T Lambda(q) Da(q)
    Lambda(q) = (Da(q)M^-1(q)Da(q)^T)^-1

    DaJ - derivative of unilateral constraint
    gamma - coefficient of restitution
    evaluate - evaluate at rho
    '''

    Id = Matrix.eye(DaJ.shape[1])
    if evaluate:
        Lambda = (DaJ@Minv@DaJ.T).subs(rho_subs).inv()
        return (Id - (1+gamma)*Minv@DaJ.T@Lambda@DaJ).subs(rho_subs)
    else:
        Lambda = (DaJ@Minv@DaJ.T).inv()
        return Id - (1+gamma)*Minv@DaJ.T@Lambda@DaJ

def R(DaJ):
    '''
    Reset Map
    R_J(q, dq) = q, \Delta_J(q,dq) dq
    '''
    delta = Delta(DaJ)
    R = Matrix([[q.T], [delta@dq.T]])
    return R


#define the force field when q is at simultaneous constraint activation
# no forces other than Coriolis
# ddq = M(q)^-1 (-C(q, dq) dq)

def ddq(dq):
    '''
    dq - 2x1 Matrix of velocity
    '''
    minv = Minv.subs(rho_subs)
    cor = -Cor.subs({dth1:dq[0, 0], dth2: dq[1, 0]}).subs(rho_subs)
    return minv@cor@dq

# define vector field for q=\rho
def V(dq):
    return Matrix.vstack(dq, ddq(dq))


###
#calculate the saltation matrices for the two (w, n) pairs

# salt = DR + (F^+ - DR F^-) Dh / Dh F^-
def salt(DR, Fplus, Fminus, Dh):
    return DR + (Fplus - DR@Fminus)@Dh / (Dh.dot(Fminus))

def Dh(Da):
    return Matrix.hstack(Da, Matrix.zeros(1, 2))


DRa1 = R(Da1).jacobian(x)
DRa2 = R(Da2).jacobian(x)

# Using notation from proof at least once differentiable

# same for both (w, n) pairs
drho0 = drho
drho2 = Delta(Matrix.vstack(Da2, Da1), evaluate=True)@drho

# w= ( {}, {}, {} )
# n1 = (1, 2)
drho1 = Delta(Da1, evaluate=True)@drho
DR1 = DRa1.subs({dq[0]:drho0[0], dq[1]:drho0[1]}).subs(rho_subs)
Xi1 = salt(DR1, V(drho1), V(drho0), Dh(Da1))
DR2 = DRa2.subs({dq[0]:drho1[0], dq[1]:drho1[1]}).subs(rho_subs)
Xi2 = salt(DR2, V(drho2), V(drho1), Dh(Da2))

#constraint 1 activated first
Xi_n1 = Xi2@Xi1

# w = ( {}, {}, {} )
# n2 = (2, 1)
drho1 = Delta(Da2, evaluate=True)@drho

DR1 = DRa2.subs({dq[0]:drho0[0], dq[1]:drho0[1]}).subs(rho_subs)
Xi1 = salt(DR1, V(drho1), V(drho0), Dh(Da2))
DR2 = DRa1.subs({dq[0]:drho1[0], dq[1]:drho1[1]}).subs(rho_subs)
Xi2 = salt(DR2, V(drho2), V(drho1), Dh(Da1))

#constraint 2 activated first
Xi_n2 = Xi2@Xi1
