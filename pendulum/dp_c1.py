from sympy import Matrix, symbols, sin, cos, acos, sqrt
from sympy.physics.mechanics import dynamicsymbols

# define the state
# q = [ \theta_1, x_2, y_2, \theta_2]
q = Matrix(dynamicsymbols('q[:4]'))
dq = Matrix(dynamicsymbols('q[:4]', 1))
th1, x2, y2, th2 = q[:]
dth1, dx2, dy2, dth2 = dq[:]
x = Matrix.vstack(q, dq)

t = symbols('t')

# paramters
l1, l2 = symbols('l_1, l_2', positive=True)
m1, m2 = symbols('m_1, m_2', positive=True)
ks = symbols('k')  #spring potential
s0 = symbols('s')  #spring nominal length


####
# mass matrix and Corrolis
I1 = 1/3 * m1 * l1**2
I2 = 1/12 * m2 * l2**2
M = Matrix.diag([I1+1/4*l1**2*m1, m2, m2, I2])
Minv = M.inv()

Cor = Matrix.zeros(2, 2)

####
# parameter values
l1_val = 1/2
l2_val = 2/7
m1_val = m2_val = 1
s0_val = 0
k_val = 100
corest = .3  # coefficient of restitution

param_vals = {l1:l1_val, l2:l2_val, m1:m1_val, m2:m2_val, s0:s0_val, ks:k_val}

Minv = Minv.subs(param_vals)

# values of the state at the simultaneous unilateral constraint activation
# chosing values similar to impact for dp_pcr with a nominal spring length of 0

th1_val = 0
th2_val = acos(-2/3)  #TODO change this!!!!
dth1_val = -1
dth2_val = 1  ##TODO change this!!!!

rot = Matrix([[cos(th1), -sin(th1)], [sin(th1), cos(th1)]])
com2 = rot@Matrix([l2/2*cos(th2), l2/2*sin(th2)])+Matrix([l1*cos(th1), l1*sin(th1)])
dcom2 = com2.diff(t).subs({th1:th1_val, th2:th2_val, dth1:dth1_val, dth2:dth2_val})
dcom2 = dcom2.subs(param_vals)
com2 = com2.subs(param_vals).subs({th1:th1_val, th2:th2_val})  # COM of rod 2

rho = Matrix([th1_val, com2[0], com2[1], th2_val])
drho = Matrix([dth1_val, dcom2[0], dcom2[1], dth2_val])

rho_subs = {_[0]:_[1] for _ in zip(q, rho)}


####
# Define the system

# Define the guards

a1 = th1 - th1_val
a2 = -th2 + th2_val

Da1 = a1.diff(q.T)
Da2 = a2.diff(q.T)

#orthogonality check
ortho_check = Da1@Minv@Da2.T
assert ortho_check[0] == 0

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
    R = Matrix([[q], [delta@dq]])
    return R


#define the force field when q is at simultaneous constraint activation
# no forces other than force due to spring

# M ddq = Fs
#Only force is from the potential in the spring
s = sqrt(((x2-l2/2*cos(th2))-l1*cos(th1))**2 + ((y2-l2/2*sin(th2))-l1*sin(th1))**2)
P = .5*ks*(s-s0)**2
Fs = P.diff(q).subs(param_vals)

def ddq(dq):
    '''
    acceleration evaluated at rho
    dq - 4x1 Matrix of velocity
    '''
    minv = Minv.subs(rho_subs)
    f = Fs.subs(rho_subs)
    return minv@f

# define vector field for q=\rho
def V(dq):
    return Matrix.vstack(dq, ddq(dq))


###
#calculate the saltation matrices for the two (w, n) pairs

# salt = DR + (F^+ - DR F^-) Dh / Dh F^-
def salt(DR, Fplus, Fminus, Dh):
    return DR + (Fplus - DR@Fminus)@Dh / (Dh.dot(Fminus))

def Dh(Da):
    return Matrix.hstack(Da, Matrix.zeros(1, len(q)))


DRa1 = R(Da1).jacobian(x)
DRa2 = R(Da2).jacobian(x)

# Using notation from proof at least once differentiable

# values same for both (w,n)
drho0 = drho
drho2 = Delta(Matrix.vstack(Da2, Da1), evaluate=True)@drho

# w = ( {}, {}, {} )
# n1 = (2, 1)

drho1 = Delta(Da2, evaluate=True)@drho
DR1 = DRa2
Dh1 = Dh(Da2)
Xi1 = salt(DR1, V(drho1), V(drho0), Dh1)
DR2 = DRa1
Dh2 = Dh(Da1)
Xi2 = salt(DR2, V(drho2), V(drho1), Dh2)
Xi_n1 = Xi2@Xi1

# w = ( {}, {}, {} )
# n1 = (1, 2)
drho1 = Delta(Da1, evaluate=True)@drho
DR1 = DRa1
Dh1 = Dh(Da1)
Xi1 = salt(DR1, V(drho1), V(drho0), Dh1)
DR2 = DRa2
Dh2 = Dh(Da2)
Xi2 = salt(DR2, V(drho2), V(drho1), Dh2)
Xi_n2 = Xi2@Xi1
