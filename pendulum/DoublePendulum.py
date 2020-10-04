# vim: expandtab tabstop=2 shiftwidth=2

import copy
import sympy
from scipy import linalg as la
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import util

from pendulum import _fig_folder

gnd_style = {'facecolor':'peachpuff', 'edgecolor':'black', 'hatch':'//', 'fill':True}
a1_color = 'red'
a2_color = 'blue'
a2_style = {'facecolor':'peachpuff', 'edgecolor':'black', 'hatch':'//'}
a1_outline = {'facecolor':'none', 'edgecolor':a1_color, 'linewidth':5}
a2_outline = {'facecolor':'none', 'edgecolor':a2_color, 'linewidth':5}
segment_params = {'linestyle':'-', 'lw':10}
marker_params = {'marker':'o', 'markersize':15, 'mec':'orange', 'mew':5, 'linestyle':'None'}

def draw_ground(ax, p, z=0, depth=.1, xc=0.0, width=6):
  '''
  Draw a patch representing the ground

  inputs:
    ax - axis to draw on
    p - parameter dict
    z - z height
    depth - how tall to draw the path
    xc - right edge of patch
    width - width of the patch
  '''

  rect = patches.Rectangle((xc-width, z-depth), width, depth, **gnd_style)
  ax.add_patch(rect)

  wedge = patches.Wedge([0, 0], depth, theta1=-90, theta2=0, zorder=-81, **gnd_style)
  ax.add_patch(wedge)

  wedge = patches.Wedge([0, 0], depth, theta1=-90, theta2=0, zorder=-80, **a1_outline)
  ax.add_patch(wedge)

  return ax

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
    a = np.array([0., 0.])
    a[0] = q[0] + 0
    a[1] = -q[1] + p['a1']
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

    #define a1 guard such that the guards are orthogonal
    b = p['m1']*p['l0']*p['l1']/2
    d = p['m1']*p['l1']**2/3+p['m1']*(p['l1']/2)**2
    p['a1'] = np.arccos(-d/b)
    return p

  @classmethod
  def calc_segments(cls, q, p):
    # calculate segment end points in complex numbers

    O = 0  # orign
    P1 = p['l0']*np.exp(1.j*q[0])
    P2 = p['l1']*np.exp(1.j*(q[0]+q[1]))+P1

    seg1 = np.array([O, P1])
    seg2 = np.array([P1, P2])

    return seg1, seg2, P1, P2

  @classmethod
  def anim(cls, trjs, p, fps=24):
    fig, ax = plt.subplots(1, 1)

    t, Q, _, _ = util.obs(trjs, DoublePendulum, p)
    q0 = Q[0]

    handles = {}

    def init():
      ax.axis('equal')
      ax.set_xlim(-.2, .7)
      ax.set_ylim(-.1, .7)
      draw_ground(ax, p)
      _, handles_fig = cls.draw_config(q0, p, ax, color='black')
      handles['seg2'] = handles_fig[0]
      handles['seg1'] = handles_fig[1]
      handles['hinges'] = handles_fig[2]
      handles['a2'] = handles_fig[3]
      handles['a2_outline'] = handles_fig[4]

      return handles.values()

    def animate(_t):
      i = (t >= _t).nonzero()[0][0]
      q = Q[i]
      seg1, seg2, P1, _ = cls.calc_segments(q, p)

      #segment 2
      handles['seg2'].set_data(seg2.real, seg2.imag)

      #segment 1
      handles['seg1'].set_data(seg1.real, seg1.imag)

      #hinges
      handles['hinges'].set_data(seg1.real, seg1.imag)

      #a2 constraint

      theta1, theta2 = np.rad2deg(q[0]+p['a1']), -180+np.rad2deg(q[0])
      handles['a2'].set_center([P1.real, P1.imag])
      handles['a2'].set_theta1(theta1)
      handles['a2'].set_theta2(theta2)
      handles['a2_outline'].set_center([P1.real, P1.imag])
      handles['a2_outline'].set_theta1(theta1)
      handles['a2_outline'].set_theta2(theta2)
      return handles.values()

    interval = int(1./fps*1000)
    ani = FuncAnimation(fig, animate, np.arange(0., t[-1], 1./fps), init_func=init,
                        blit=True, repeat=True, interval=200)

    plt.ion()
    plt.show()

    return ani


  @classmethod
  def draw_config(cls, q, p, ax=None, color='blue'):
    '''Draw the configuration of the double pendulum

    q - [theta0, theta1]
    p - parameter dict with l0 and l1
    ax - axis to draw the parameter on, if None create new fig and ax

    additional parameters in p change the color of the plotted lines:
      constraint_line_param = {'color':.8}
      beam_line_param = {'color': 'blue'}

    return ax
    '''
    if ax is None:
      _, ax = plt.subplots(1)

    seg1, seg2, P1, _ = cls.calc_segments(q, p)

    #plot segments
    seg2_handle, = ax.plot(seg2.real, seg2.imag, **segment_params, color=color)
    seg1_handle,= ax.plot(seg1.real, seg1.imag, **segment_params, color=color)
    #plot "hinges"
    hinges_handle, = ax.plot(seg1.real, seg1.imag, **marker_params, mfc=color)

    #plot constraints
    draw_ground(ax, p)

    # add a2
    # Two wedges as setting linecolor changes hatches color
    wedge = patches.Wedge([P1.real, P1.imag], .15, np.rad2deg(q[0]+p['a1']), -180+np.rad2deg(q[0]),
                          **a2_style, zorder=-100)
    ax.add_patch(wedge)
    wedge_outline = patches.Wedge([P1.real, P1.imag], .15, np.rad2deg(q[0]+p['a1']), -180+np.rad2deg(q[0]),
        zorder=-99, **a2_outline)
    ax.add_patch(wedge_outline)

    handles = [seg2_handle, seg1_handle, hinges_handle, wedge, wedge_outline]

    return ax, handles


def generate_DP_impages():
  plt.ion()
  p = DoublePendulum.nominal_parameters()
  rho = DoublePendulum.simultaneous_impact_configuration(p)

  t = .7
  q0, dq0 = DoublePendulum.x0(t, p)

  def set_lim(ax):
    ax.axis('equal')
    ax.set(xlim=(-.2,.7), ylim=(-.1,.7))
    ax.axis('off')

  plt.ion()


def test_anim():
  p = DoublePendulum.nominal_parameters()
  J = [0, 0]
  dt = 1e-4  # at dt=1e-3 simulatenous impact doesn't occur
  dt = 1e-3  # still see the PCr nature in perturbation plots
  rx = 1e-7
  # find initial condition for simultaneous contact
  q0 = DoublePendulum.simultaneous_impact_configuration(p)
  dq0 = DoublePendulum.simultaneous_impact_velocity()
  t0_back = .7

  ##simulate backwards
  #tstop_back = 0
  #trjs_back = util.sim(DoublePendulum, tstop_back, dt, rx, t0_back, q0, dq0, J, p)

  #q0 = trjs_back[-1]['q'][-1]
  #dq0 = trjs_back[-1]['dq'][-1]
  tstop = 2*t0_back
  t0 = 0

  q0 = np.array([0.68154256, 1.29426896])
  dq0 = np.array([-0.99205791, 1.9429129])
  trjs = util.sim(DoublePendulum, tstop, dt, rx, t0, q0, dq0, J, p)
  ani = DoublePendulum.anim(trjs, p)

if __name__ == '__main__':
  test_anim()
  #plt.ion()
  #p = DoublePendulum.nominal_parameters()

  #t = .7
  #q0, dq0 = DoublePendulum.x0(t, p)
  #def set_lim(ax):
  #  ax.axis('equal')
  #  ax.set(xlim=(-.2,.7), ylim=(-.1,.7))
  #  ax.axis('off')

  #plt.ion()

  #fig, ax = plt.subplots(1)
  #DoublePendulum.draw_config(q0, p, ax=ax, color='black')
  #set_lim(ax)
  #plt.savefig(_fig_folder / 'nominal.svg')
