# vim: expandtab tabstop=2 shiftwidth=2
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from sympy.physics.mechanics import dynamicsymbols

from hybrid import HybridSystem

#parameters for drawings / animations
body_color = 'k'
left_color = 'b'
right_color = 'r'
body_style = {'linestyle':'-', 'lw':10, 'color':body_color}
leg_style = {'linestyle':'-', 'lw':5, 'color':'grey'}
spring_leg_style = {'linestyle':':', 'lw':5, 'color':'grey'}
foot_style = {'marker':'o', 'markersize':15}

def Rot(theta):
  R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
  return R

class Biped(HybridSystem):
  N_States = 7

  # state indices q = [xb xl xr zb zl zr theta]
  ixb, ixl, ixr, izb, izl, izr, ith = np.arange(N_States)

  @classmethod
  def str(cls):
    return "Biped"

  @classmethod
  def ic(cls, p, theta=0, height=1.):
    '''
    input:
    p - parameter dictionary
    theta - initial rotation CCW from z-axis
    height - initial height of body center (top)

    output:
      q, dq,  - initial configuration and velocity vectors
      J       - initial discrete state
    '''
    assert height > 0

    q = np.zeros(cls.N_States)
    dq = np.zeros(cls.N_States)
    q[cls.ixl] = -p['wh']
    q[cls.ixr] = p['wh']
    q[cls.izl] = q[cls.izr] = -p['l']

    #rotation
    left = np.exp(1j*theta)*(q[cls.ixl]+1j*q[cls.izl])
    right = np.exp(1j*theta)*(q[cls.ixr]+1j*q[cls.izr])
    q[cls.ixl] = np.real(left)
    q[cls.izl] = np.imag(left)
    q[cls.ixr] = np.real(right)
    q[cls.izr] = np.imag(right)

    q[cls.ith] = theta

    #height
    q[[cls.izl, cls.izr, cls.izb]] += height

    J = np.array([0, 0])  # initial discrete state

    return q, dq, J

  @classmethod
  def M(cls, t, k, q, J, p):
    mb = p['mb']
    mf = p['mf']
    Ib = p['Ib']

    M = np.diag([mb, mf, mf, mb, mf, mf, Ib])
    return M

  @classmethod
  def nominal_parameters(cls):
    p = {}
    p['mb'] = 1
    p['mf'] = 1
    p['Ib'] = 1
    p['g'] = 1
    p['l'] = 1/2.
    p['wh'] = p['l']/2.
    p['gamma'] = 0  # perfectly plastic impacts
    return p

  #  unilateral constraint
  @classmethod
  def a(cls, t, k, q, J, p):
    a = np.array([q[cls.izl], q[cls.izr]])
    return a

  @classmethod
  def Da(cls, t, k, q, J, p):
    Da = []
    if J[0]:
      tmp = np.zeros(cls.N_States)
      tmp[cls.izl] = 1
      Da.append(tmp)
    if J[1]:
      tmp = np.zeros(cls.N_States)
      tmp[cls.izr] = 1
      Da.append(tmp)

    Da = np.array(Da)
    if Da.size > 0:
      return Da
    else:
      return np.array([]).reshape(0, len(q))

  @classmethod
  def DtDa(cls, t, k, q, dq, J, p):
    return np.zeros((np.sum(J), len(q)))

  @classmethod
  def Db(cls, t, k, q, J, p):
    return np.array([]).reshape(0, len(q))

  @classmethod
  def DtDb(cls, t, k, q, dq, J, p):
    return np.array([]).reshape(0, len(q))


class RigidBiped(Biped):
  DxF = None
  Db_sym = None
  DtDb_sym = None

  @classmethod
  def str(cls):
    return "Rigid Biped"

  # derivative of bilateral constraints
  # constraints a3-a6 in the corresponding paper
  @classmethod
  def gen_bilateral_constraints(cls):
    q = sym.Matrix(dynamicsymbols('q[:7]'))
    dq = sym.Matrix(dynamicsymbols('q[:7]', 1))
    t = sym.symbols('t')

    theta = q[cls.ith]

    R = sym.Matrix([[sym.cos(-theta), -sym.sin(-theta)],
                    [sym.sin(-theta), sym.cos(-theta)]])

    br = R@(q[[cls.ixr, cls.izr], 0] - q[[cls.ixb, cls.izb], 0])

    bl = R@(q[[cls.ixl, cls.izl], 0] - q[[cls.ixb, cls.izb], 0])
    b = sym.Matrix.vstack(br, bl)
    # there are constants in the bilateral constraint
    # which are removed with derivative
    Db = b.jacobian(q)
    DtDb = Db.diff(t)

    cls.Db_sym = sym.lambdify([q], Db)
    cls.DtDb_sym = sym.lambdify([q, dq], DtDb)

  @classmethod
  def Db(cls, t, k, q, J, p):
    if cls.Db_sym is None or cls.DtDb_sym is None:
      cls.gen_bilateral_constraints()
    return np.asarray(cls.Db_sym(q)).astype(np.float64)

  @classmethod
  def DtDb(cls, t, k, q, dq, J, p):
    if cls.Db_sym is None or cls.DtDb_sym is None:
      cls.gen_bilateral_constraints()
    return np.asarray(cls.DtDb_sym(q, dq)).astype(np.float64)

  @classmethod
  def f(cls, t, k, q, dq, J, p):
    g = np.zeros(cls.N_States)
    g[[cls.izb, cls.izr, cls.izl]] = -p['g']

    fg = cls.M(t, k, q, J, p)@g
    return fg

  @classmethod
  def draw_config(cls, q, p, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1)

    hipl = np.exp(1j*q[cls.ith])*(-p['wh']) + (q[cls.ixb] + 1j*q[cls.izb])
    hipr = np.exp(1j*q[cls.ith])*(+p['wh']) + (q[cls.ixb] + 1j*q[cls.izb])
    body = [hipl, hipr]
    lleg = [hipl, (q[cls.ixl]+1j*q[cls.izl])]
    rleg = [hipr, (q[cls.ixr]+1j*q[cls.izr])]

    body_color = 'k'
    left_color = 'b'
    right_color = 'r'
    body_style = {'linestyle':'-', 'lw':10, 'color':body_color}
    leg_style = {'linestyle':'-', 'lw':5, 'color':'grey'}
    foot_style = {'marker':'o', 'markersize':15}

    ax.plot(np.real(body), np.imag(body), **body_style)
    ax.plot(np.real(lleg), np.imag(lleg), **leg_style, zorder=-1)
    ax.plot(np.real(rleg), np.imag(rleg), **leg_style, zorder=-1)
    ax.plot(q[cls.ixl], q[cls.izl], **foot_style, color=left_color)
    ax.plot(q[cls.ixr], q[cls.izr], **foot_style, color=right_color)

    return ax

class DecoupledBiped(Biped):
  Fs = None
  DxF = None

  @classmethod
  def str(cls):
    return "Decoupled Biped"

  @classmethod
  def nominal_parameters(cls):
    p = super(DecoupledBiped, cls).nominal_parameters()
    p['k'] = 27  # spring constant
    # using p['l'] as nominal spring length
    return p

  @classmethod
  def define_spring_force(cls, p):
    # Assumes once parameter is set, the values never change
    qsym = sym.symarray('q', cls.N_States, real=True)
    theta = qsym[cls.ith]
    xb, xl, xr, zb, zl, zr = qsym[[cls.ixb, cls.ixl, cls.ixr,
                                   cls.izb, cls.izl, cls.izr]]

    R = sym.Matrix([[sym.cos(theta), -sym.sin(theta)],
                    [sym.sin(theta), sym.cos(theta)]])

    body = sym.Matrix([[xb], [zb]])
    footl = sym.Matrix([[xl], [zl]])
    footr = sym.Matrix([[xr], [zr]])

    #length of right spring
    hipr = R@sym.Matrix([[p['wh']], [0]]) + body
    hipl = R@sym.Matrix([[-p['wh']], [0]]) + body

    #spring length
    springr = (hipr - footr).norm()
    springl = (hipl - footl).norm()

    #potential energy from the springs
    P = .5*p['k']*(springr - p['l'])**2 + \
        .5*p['k']*(springl - p['l'])**2
    # L = T - V
    # potential force = dL/dx = -(partial/partial q) P
    Fs = -P.diff(qsym)

    cls.Fs = sym.lambdify([qsym], Fs)

  @classmethod
  def f(cls, t, k, q, dq, J, p):
    if cls.Fs is None:
      cls.define_spring_force(p)

    g = np.zeros(cls.N_States)
    g[[cls.izb, cls.izr, cls.izl]] = -p['g']
    fs = cls.Fs(q)

    fg = cls.M(t, k, q, J, p)@g
    return fg + fs

  @classmethod
  def draw_config(cls, q, p, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1)

    hipl = np.exp(1j*q[cls.ith])*(-p['wh']) + (q[cls.ixb] + 1j*q[cls.izb])
    hipr = np.exp(1j*q[cls.ith])*(+p['wh']) + (q[cls.ixb] + 1j*q[cls.izb])
    body = [hipl, hipr]
    lleg = [hipl, (q[cls.ixl]+1j*q[cls.izl])]
    rleg = [hipr, (q[cls.ixr]+1j*q[cls.izr])]

    ax.plot(np.real(body), np.imag(body), **body_style)
    ax.plot(np.real(lleg), np.imag(lleg), **spring_leg_style, zorder=-1)
    ax.plot(np.real(rleg), np.imag(rleg), **spring_leg_style, zorder=-1)
    ax.plot(q[cls.ixl], q[cls.izl], **foot_style, color=left_color)
    ax.plot(q[cls.ixr], q[cls.izr], **foot_style, color=right_color)

    return ax


class PCrBiped(DecoupledBiped):
  @classmethod
  def str(cls):
    return "PCr Biped"

  @classmethod
  def nominal_parameters(cls):
    p = super(PCrBiped, cls).nominal_parameters()
    p['b'] = 1  #actuated flywheel forces
    return p

  @classmethod
  def f(cls, t, k, q, dq, J, p):
    #conservative forces from potential energy (spring and gravity)
    fp = super(PCrBiped, cls).f(t, k, q, dq, J, p)

    f_fly = np.zeros(len(q))
    f_fly[cls.ith] = p['b']*(dq[cls.izl] - dq[cls.izr])**2
    f = fp + f_fly
    return f


if __name__ == '__main__':
  p = RigidBiped.nominal_parameters()
  q0, dq0, J = RigidBiped.ic(p, .1)
  DecoupledBiped.draw_config(q0, p)
