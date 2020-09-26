# vim: expandtab tabstop=2 shiftwidth=2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sympy as sym
from sympy.physics.mechanics import dynamicsymbols

from hybrid import HybridSystem

#parameters for drawings / animations
body_color = 'k'
left_color = 'b'
right_color = 'r'
default_leg_style = {'linestyle':'-', 'lw':5, 'color':'grey'}
default_gnd_style = {'facecolor':'brown', 'edgecolor':'black', 'hatch':'/', 'fill':True}

default_flywheel_style = {'marker':'o', 'markersize':20, 'color':'darkgrey',
                          'markeredgecolor':'dimgrey', 'mew':3}
default_foot_style = {'marker':'o', 'markersize':10}
default_spring_leg_style = {'lw':3, 'color':'grey'}
default_body_style = {'linestyle':'-', 'lw':10, 'color':'black'}

def Rot(theta):
  R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
  return R

def draw_ground(ax, p, z=0, depth=.1, xc=0, width=1):
  '''
  Draw a patch representing the ground

  inputs:
    ax - axis to draw on
    p - parameter dict
    z - z height
    depth - how tall to draw the path
    xc - center of path
    width - width of the patch
  '''

  if 'gnd_style' in p:
    gnd_style = p['gnd_style']
  else:
    gnd_style = default_gnd_style

  rect = patches.Rectangle((xc-width/2, z-depth), width, depth, **gnd_style)
  ax.add_patch(rect)
  return ax


class Biped(HybridSystem):
  N_States = 7
  DxF_lam = defaultdict(lambda: None)  #need multipled, one for each J

  # state indices q = [xb xl xr zb zl zr theta]
  ixb, ixl, ixr, izb, izl, izr, ith = np.arange(N_States)

  @classmethod
  def DxF(cls, t, k, q, dq, J, p):
    if cls.DxF_lam[tuple(J)] is None:
      p = p.copy()
      p['symbolic'] = True
      #q_sym = sym.symbols('q[:'+str(cls.N_States)+']')
      #dq_sym = sym.symbols('dq[:'+str(cls.N_States)+']')
      q_sym = sym.symarray('q', cls.N_States, real=True)
      dq_sym = sym.symarray('dq', cls.N_States, real=True)

      # k, t, J don't change the calculations
      ddq_sym = sym.Matrix(cls.ddq(0, 0, q_sym, dq_sym, J, p))
      vec_field = sym.Matrix.vstack(sym.Matrix(dq_sym), ddq_sym)

      DxF_sym = vec_field.jacobian(sym.Matrix([*q_sym, *dq_sym]))
      DxF_lam = sym.lambdify([q_sym, dq_sym], DxF_sym)

      cls.DxF_lam[tuple(J)] = DxF_lam

    return np.asarray(cls.DxF_lam[tuple(J)](q, dq)).astype(np.float64)

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
    p['quick_observ'] = True  #calculate only minimal state values when returning O
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
  def R(cls, t, k, q, dq, J, p):
    '''
    The RigidBiped behaves like the rocking block,
    unless both unilateral constraints activate at the same instant,
    the activation of one constraint causes the other to become inactive

    Always assuming plastic collision
    '''
    assert p['gamma'] == 0

    a = cls.a(t, k, q, J, p)
    Jimpact = a < 0
    if np.sum(J) == 1:  #that is one constraint is active
      Jnew = np.logical_not(J)
    else:
      Jnew = Jimpact

    Delta = cls.Delta(t, k, q, dq, Jnew, p)
    dq_ = Delta@dq

    return q, dq_, Jnew

  # derivative of bilateral constraints
  # constraints a3-a6 in the corresponding paper
  @classmethod
  def _gen_bilateral_constraints(cls):
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
      cls._gen_bilateral_constraints()
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

    if 'body_style' in p:
      body_style = p['body_style']
    else:
      body_style = default_body_style

    if 'leg_style' in p:
      leg_style = p['leg_style']
    else:
      leg_style = default_leg_style

    if 'foot_style' in p:
      foot_style = p['foot_style']
    else:
      foot_style = default_foot_style

    ax.plot(np.real(body), np.imag(body), **body_style)
    ax.plot(np.real(lleg), np.imag(lleg), **leg_style, zorder=-1)
    ax.plot(np.real(rleg), np.imag(rleg), **leg_style, zorder=-1)
    ax.plot(q[cls.ixl], q[cls.izl], **foot_style, color=left_color)
    ax.plot(q[cls.ixr], q[cls.izr], **foot_style, color=right_color)

    ax = draw_ground(ax, p)
    return ax

class DecoupledBiped(Biped):
  Fs = None

  @classmethod
  def nominal_parameters(cls):
    p = super(DecoupledBiped, cls).nominal_parameters()
    p['k'] = 27  # spring constant
    # using p['l'] as nominal spring length
    return p

  @classmethod
  def define_spring_force(cls, p, qsym=None):
    # Assumes once parameter is set, the values never change
    if qsym is None:
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

    if 'symbolic' in p and p['symbolic']:
      return Fs

    cls.Fs = sym.lambdify([qsym], Fs)

  @classmethod
  def f(cls, t, k, q, dq, J, p):
    if 'symbolic' in p and p['symbolic']:
      Fs = cls.define_spring_force(p, q)
      g = np.zeros(cls.N_States)
      g[[cls.izb, cls.izr, cls.izl]] = -p['g']
      fg = cls.M(t, k, q, J, p)@g

      return sym.Array(Fs) + sym.Array(fg)

    if cls.Fs is None:
      cls.define_spring_force(p)

    g = np.zeros(cls.N_States)
    g[[cls.izb, cls.izr, cls.izl]] = -p['g']
    fs = cls.Fs(q)

    fg = cls.M(t, k, q, J, p)@g
    return fg + fs

  @classmethod
  def spring(cls, z0, z1, a=.2, b=.6, c=.2, h=1., p=4, N=100):
    '''
    Generate coordinates for a spring (in imaginary coordinates)

    From - Sam Burden

    input:
      z0, z1 - initial/ final coordinates (imaginary)
      a, b, c - relative ratio of inital straight to squiggle to final straight
                line segments
      h - width of squiggle
      N - number of points
    '''

    x = np.linspace(0., a+b+c, N)
    y = 0.*x
    mb = int(N*a/(a+b+c))
    Mb = int(N*(a+b)/(a+b+c))
    y[mb:Mb] = np.mod(np.linspace(0., p-.01, Mb-mb), 1.)-0.5
    z = ((np.abs(z1-z0)*x + 1.j*h*y)) * np.exp(1.j*np.angle(z1-z0)) + z0
    return z

  @classmethod
  def draw_config(cls, q, p, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1)

    hipl = np.exp(1j*q[cls.ith])*(-p['wh']) + (q[cls.ixb] + 1j*q[cls.izb])
    hipr = np.exp(1j*q[cls.ith])*(+p['wh']) + (q[cls.ixb] + 1j*q[cls.izb])
    body = [hipl, hipr]
    lleg = [hipl, (q[cls.ixl]+1j*q[cls.izl])]
    rleg = [hipr, (q[cls.ixr]+1j*q[cls.izr])]

    if 'body_style' in p:
      body_style = p['body_style']
    else:
      body_style = default_body_style

    if 'spring_leg_style' in p:
      spring_leg_style = p['spring_leg_style']
    else:
      spring_leg_style = default_spring_leg_style

    if 'foot_style' in p:
      foot_style = p['foot_style']
    else:
      foot_style = default_foot_style

    ax.plot(np.real(body), np.imag(body), **body_style)
    #ax.plot(np.real(lleg), np.imag(lleg), **spring_leg_style, zorder=-1)
    #ax.plot(np.real(rleg), np.imag(rleg), **spring_leg_style, zorder=-1)
    lleg_spring = cls.spring(lleg[0], lleg[1], h=.1)
    rleg_spring = cls.spring(rleg[0], rleg[1], h=.1)
    ax.plot(np.real(lleg_spring), np.imag(lleg_spring), **spring_leg_style, zorder=-1)
    ax.plot(np.real(rleg_spring), np.imag(rleg_spring), **spring_leg_style, zorder=-1)
    ax.plot(q[cls.ixl], q[cls.izl], **foot_style, color=left_color)
    ax.plot(q[cls.ixr], q[cls.izr], **foot_style, color=right_color)

    ax = draw_ground(ax, p)
    return ax


class PCrBiped(DecoupledBiped):
  @classmethod
  def draw_config(cls, q, p, ax=None):
    ax = super(PCrBiped, cls).draw_config(q, p, ax)

    #draw the flywheel
    if 'flywheel_style' in p:
      ax.plot(q[cls.ixb], q[cls.izb], **p['flywheel_style'])
      ax.plot(q[cls.ixb], q[cls.izb], marker='.', color='black', markersize=5)

    else:
      ax.plot(q[cls.ixb], q[cls.izb], **default_flywheel_style)
      ax.plot(q[cls.ixb], q[cls.izb], marker='.', color='black', markersize=5)
    return ax

  @classmethod
  def nominal_parameters(cls):
    p = super(PCrBiped, cls).nominal_parameters()
    p['b'] = 1.5  #actuated flywheel forces
    return p

  @classmethod
  def f(cls, t, k, q, dq, J, p):
    #conservative forces from potential energy (spring and gravity)
    fp = super(PCrBiped, cls).f(t, k, q, dq, J, p)

    f_fly = np.zeros(len(q))
    if 'symbolic' in p and p['symbolic']:
      f_fly = sym.zeros(len(q), 1)

    f_fly[cls.ith] = p['b']*(dq[cls.izl] - dq[cls.izr])**2

    if 'symbolic' in p and p['symbolic']:
      f_fly = sym.Array(f_fly, len(q),)
    f = fp + f_fly
    return f


if __name__ == '__main__':
  plt.ion()
  p = RigidBiped.nominal_parameters()
  q0, dq0, J = RigidBiped.ic(p, .1)
  ax = PCrBiped.draw_config(q0, p)
  ax.axis('off')
  plt.tight_layout()
