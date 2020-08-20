# vim: expandtab tabstop=2 shiftwidth=2

import copy
import sympy
from scipy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

import util

class DoublePendulum:
  DxF_lam = None

  @classmethod
  def M(cls, t, k, q, J, p):
    m0 = p['m0']
    m1 = p['m1']
    l0 = p['l0']
    l1 = p['l1']
    a = m0*l0**2/3 + m1*l1**2/3 + m0*(l0/2)**2 + m1*(l0**2+(l1/2)**2)
    b = m1*l0*l1/2
    d = m1*l1**2/3+m1*(l1/2)**2
    if 'symbolic' in p and p['symbolic']:
      cos = sympy.cos
    else:
      cos = np.cos

    m00 = a+2*b*cos(q[1])
    m10 = d+b*cos(q[1])
    m11 = d

    M = np.array([[m00, m10], [m10, m11]])
    return M

  @classmethod
  def O(cls, t, k, q, dq, J, p):
    o = {}
    o['q'] = q
    o['dq'] = dq
    o['g'] = cls.G(t, k, q, dq, J, p)
    return o

  @classmethod
  def C(cls, t, k, q, dq, J, p):
    #m0 = p['m0']
    m1 = p['m1']
    l0 = p['l0']
    l1 = p['l1']
    #a = m0*l0**2/3 + m1*l1**2/3 + m0*(l0/2)**2 + m1*(l0**2+(l1/2)**2)
    b = m1*l0*l1/2
    #d = m1*l1**2/3+m1*(l1/2)**2
    if 'symbolic' in p and p['symbolic']:
      sin = sympy.sin
    else:
      sin = np.sin
    c00 = -b*sin(q[1])*dq[1]
    c01 = -b*sin(q[1])*(dq[0]+dq[1])
    c10 = b*sin(q[1])*dq[0]
    c11 = 0

    C = np.array([[c00, c01], [c10, c11]])
    return C

  @classmethod
  @util.mechanics_memoization
  def ddq(cls, t, k, q, dq, J, p):
    '''
    No external forces
    M ddq + C(q, dq) dq = 0
    '''
    M = cls.M(t, k, q, J, p)
    if 'symbolic' in p and p['symbolic']:
      Minv = sympy.Matrix(M).inv()
    else:
      # not the numerically correct way to solve the equation ... oh well
      Minv = la.inv(M)
    ddq = Minv@(-cls.C(t, k, q, dq, J, p)@dq)
    return ddq

  @classmethod
  def DxF(cls, t, k, q, dq, J, p):
    if cls.DxF_lam is None:
      p['symbolic'] = True
      q_sym = sympy.symbols('q[:2]')
      dq_sym = sympy.symbols('dq[:2]')

      # k,t,J don't change ddq calculation
      ddq_sym = sympy.Matrix(cls.ddq(0, 0, q_sym, dq_sym, [0, 0], p))
      vec_field = sympy.Matrix.vstack(sympy.Matrix(dq_sym), ddq_sym)

      DxF_sym = vec_field.jacobian(sympy.Matrix([*q_sym, *dq_sym]))
      DxF_lam = sympy.lambdify([q_sym, dq_sym], DxF_sym)

      cls.DxF_lam = DxF_lam

    return np.asarray(cls.DxF_lam(q, dq)).astype(np.float64)

  @classmethod
  def G(cls, t, k, q, dq, J, p):
    g = cls.a(t, k, q, dq, J, p)
    return g

  #  unilateral constraint
  @classmethod
  def a(cls, t, k, q, dq, J, p):
    b = p['m1']*p['l0']*p['l1']/2
    d = p['m1']*p['l1']**2/3+p['m1']*(p['l1']/2)**2
    a = np.array([0., 0.])
    a[0] = q[0] + 0
    a[1] = -q[1] + np.arccos(-d/b)
    return a

  @classmethod
  def Da(cls, t, k, q, dq, J, p):
    Da = []
    if J[0]:
      Da.append([1, 0])
    if J[1]:
      Da.append([0, -1])
    Da = np.array(Da)
    if Da.size > 0:
      return Da
    else:
      return np.array([]).reshape(0, len(q))
    return Da

  @classmethod
  def Delta(cls, t, k, q, dq, J, p):
    d = len(q)
    gamma = p['gamma']
    M = cls.M(t, k, q, J, p)
    Da = cls.Da(t, k, q, dq, J, p)
    Lambda = -la.inv(Da@la.inv(M)@Da.T)

    Delta = np.identity(d) + (1+gamma)*la.inv(M)@Da.T@Lambda@Da
    return Delta

  @classmethod
  def R(cls, t, k, q, dq, J, p):
    '''
    non plastic impact
    '''
    a = cls.a(t, k, q, dq, J, p)
    Jimpact = a < 0
    Delta = cls.Delta(t, k, q, dq, Jimpact, p)
    dq_ = Delta@dq
    return q, dq_, copy.copy(J)

  @classmethod
  def simultaneous_impact_configuration(cls, p):
    b = p['m1']*p['l0']*p['l1']/2
    d = p['m1']*p['l1']**2/3+p['m1']*(p['l1']/2)**2
    q0 = np.array([0, np.arccos(-d/b)])
    return q0

  @classmethod
  def simultaneous_impact_velocity(cls):
    return np.array([-1, 1])

  @classmethod
  def x0(cls, t, p):
    '''
    t - time till simultaneous impact
    p - dictionary of parameters
    '''
    J = [0, 0]
    dt = 1e-3
    rx = 1e-7
    t0 = 0

    rho = cls.simultaneous_impact_configuration(p)
    drho = cls.simultaneous_impact_velocity()
    tstop = -t
    trjs = util.sim(cls, tstop, dt, rx, t0, rho, drho, J, p)
    q0 = trjs[-1]['q'][-1]
    dq0 = trjs[-1]['dq'][-1]

    return q0, dq0

  @classmethod
  def nominal_parameters(cls):
    p = {'m0':1, 'm1':1, 'l0':.5, 'l1':2/7}
    p['gamma'] = .3
    return p

  @classmethod
  def draw_config(cls, q, p, draw_a1=True, ax=None, color=None):
    '''Draw the configuration of the double pendulum

    q - [theta0, theta1]
    p - parameter dict with l0 and l1, (and m1 if draw_a1 is True)
    draw_a1 - Draw constraint a1
    ax - axis to draw the parameter on, if None create new fig and ax

    additional parameters in p change the color of the plotted lines:
      constraint_line_param = {'color':.8}
      beam_line_param = {'color': 'blue'}

    return ax
    '''

    default_constraint_lp = {'color':'.8'}
    if color is None:
      default_beam_lp = {'color': 'blue'}
    else:
      default_beam_lp = {'color': color}
    if ax is None:
      _, ax = plt.subplots(1)

    def R(theta):
      #Rotation matrix about the z-axis
      R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
      return R

    O = 0  # orign
    P1 = p['l0']*np.exp(1.j*q[0])
    P2 = p['l1']*np.exp(1.j*(q[0]+q[1]))+P1

    if draw_a1:
      b = p['m1']*p['l0']*p['l1']/2
      d = p['m1']*p['l1']**2/3+p['m1']*(p['l1']/2)**2
      theta = np.arccos(-d/b)
      PC = R(q[0])@np.array([np.cos(theta)*p['l1']/2, np.sin(theta)*p['l1']/2]) + P1

      lp = p.get('constraint_line_param', default_constraint_lp)
      linec = mlines.Line2D([P1[0], PC[0]], [P1[1], PC[1]], **lp)
      ax.add_line(linec)

    if color is None:
      lc = 'blue'
    else:
      lc = color
    mec = 'orange'  #marker edge color
    plot_params = {'marker':'o', 'linestyle':'-', 'markersize':15,
                   'lw':10, 'mec':mec, 'mew':5, 'mfc':lc, 'color':lc}
    seg1 = np.array([O, P1])
    seg2 = np.array([P1, P2])
    ax.plot(seg1.real, seg1.imag, **plot_params)
    ax.plot(seg2.real, seg2.imag, **plot_params)

    #lp = p.get('beam_line_param', default_beam_lp)
    #line0 = mlines.Line2D([O[0], P1[0]], [O[1], P1[1]], **lp)
    #line1 = mlines.Line2D([P1[0], P2[0]], [P1[1], P2[1]], **lp)
    #ax.add_line(line0)
    #ax.add_line(line1)

    return ax


if __name__ == '__main__':
  plt.ion()
  p = DoublePendulum.nominal_parameters()
  rho = DoublePendulum.simultaneous_impact_configuration(p)

  t = .7
  q0, dq0 = DoublePendulum.x0(t, p)

  def set_lim(ax):
    ax.axis('equal')
    ax.set(xlim=(-.2,.7), ylim=(-.1,.7))

  plt.ion()

  fig, ax = plt.subplots(1)
  DoublePendulum.draw_config(rho, p, ax=ax, draw_a1=False, color='black')
  set_lim(ax)
  ax.set_title('Configuration at simultaneous impact')
  fig.savefig('sim_impact.png')

  fig, ax = plt.subplots(1)
  DoublePendulum.draw_config(q0, p, ax=ax, draw_a1=False, color='black')
  set_lim(ax)
  ax.set_title('Configuration at t=0')
  plt.savefig('q0.png')

  #fig, ax = plt.subplots(1)
  q0plus = q0 + np.array([1, 0]) * .05
  DoublePendulum.draw_config(q0plus, p, ax=ax, draw_a1=False)
  set_lim(ax)
  plt.savefig('q0plus.png')


  fig, ax = plt.subplots(1)
  q0minus = q0 + np.array([1, 0]) * -.05
  DoublePendulum.draw_config(q0, p, ax=ax, draw_a1=False, color='black')
  DoublePendulum.draw_config(q0minus, p, ax=ax, draw_a1=False, color='red')
  set_lim(ax)
  plt.savefig('q0minus.png')
