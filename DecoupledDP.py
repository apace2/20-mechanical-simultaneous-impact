# vim: expandtab tabstop=2 shiftwidth=2

import copy
import sympy
import sympy.physics.mechanics
from scipy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import util

def Rot(theta):
  #Rotation matrix about the z-axis
  R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
  return R

class DecoupledDP():
  Fs = None  #spring force function
  DxF_lam = None  #DxF lambda function
  DOF = 4  #degrees of freedom

  @classmethod
  def M(cls, t, k, q, J, p, symbolic=False):
    I1 = 1/3 * p['m1'] * p['l1']**2
    I2 = 1/12*p['m2']*p['l2']**2
    M = np.diag([I1+1/4*p['l1']**2*p['m1'], p['m2'], p['m2'], I2])
    return M

  @classmethod
  def O(cls, t, k, q, dq, J, p):
    o = {}
    o['q'] = q
    o['dq'] = dq
    o['g'] = cls.G(t, k, q, dq, J, p)
    return o

  @classmethod
  def C(cls, t, k, q, dq, J, p, symbolic=False):
    return np.zeros((len(q), len(q)))

  @classmethod
  def symbolic_spring_force(cls, q, p):
      th1, x2, y2, th2 = q
      s = sympy.sqrt(((x2-p['l2']/2*sympy.cos(th2))-p['l1']*sympy.cos(th1))**2 +
         ((y2-p['l2']/2*sympy.sin(th2))-p['l1']*sympy.sin(th1))**2)
      P = .5*p['ks']*(s-p['s0'])**2
      Fs = P.diff(q)
      return Fs

  @classmethod
  def ddq(cls, t, k, q, dq, J, p, symbolic=False):
    '''
    Only external force due to spring
    Unilateral constraints are not active for a non instantaneous length of time

    M ddq = Fs
    '''
    if not symbolic:
      assert q.dtype != np.object

    M = cls.M(t, k, q, J, p, symbolic)
    if symbolic:
      Minv = sympy.Matrix(M).inv()
    else:
      Minv = la.inv(M)

    if cls.Fs is None:
      #qsym = sympy.symarray('q', cls.DOF)
      #Fs = cls.symbolic_spring_force(qsym, p)
      ##qsym = np.asarray(qsym)
      #cls.Fs = sympy.lambdify((qsym,), Fs)

      qsym = sympy.physics.mechanics.dynamicsymbols('q[:4]')
      th1, x2, y2, th2 = qsym
      s = sympy.sqrt(((x2-p['l2']/2*sympy.cos(th2))-p['l1']*sympy.cos(th1))**2 +
         ((y2-p['l2']/2*sympy.sin(th2))-p['l1']*sympy.sin(th1))**2)
      P = .5*p['ks']*(s-p['s0'])**2
      Fs = P.diff(sympy.Matrix(qsym))
      cls.Fs = sympy.lambdify((qsym,), Fs)

    if symbolic:
      f, _ = cls.symbolic_spring_force(p)
    else:
      f = cls.Fs(q)
    ddq = Minv@f
    return np.squeeze(ddq)

  @classmethod
  def DxF(cls, t, k, q, dq, J, p):
    # Given all methods are classmethods
    # should include p as an input
    # Challenge: symbolic matrix inverses take longer
    if DxF_lam is None:
      q_sym = sympy.symarray('q', cls.DOF)
      dq_sym = sympy.symarray('dq', cls.DOF)
      ddq_sym = cls.ddq(None, None, q_sym, dq_sym, None, p, symbolic=True)

      vec_field = sympy.Matrix.vstack( dq_sym, ddq_sym)
      DxF_sym = vec_field.jacobian(sympy.Matrix([*q_sym, *dq_sym]))
      DxF_lam = sympy.lambdify([q_sym, dq_sym])
    return DxF(q, dq)

  @classmethod
  def G(cls, t, k, q, dq, J, p):
    g = cls.a(t, k, q, dq, J, p)
    return g

  #  unilateral constraint
  @classmethod
  def a(cls, t, k, q, dq, J, p):
    rho = cls.simultaneous_impact_configuration(p)

    a = np.array([0, 0], dtype=np.float64)
    a[0] = q[0] - rho[0]
    a[1] = -q[3] + rho[3]
    return a

  @classmethod
  def Da(cls, t, k, q, dq, J, p):
    Da = []
    if J[0]:
      Da.append([1, 0, 0, 0])
    if J[1]:
      Da.append([0, 0, 0, -1])
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
    th1 = 0
    th2 = np.arccos(-2/3)
    rot = Rot(th1)
    com2 = rot@np.array([[p['l2']/2*np.cos(th2)], [p['l2']/2*np.sin(th2)]]) + \
        np.array([[p['l1']*np.cos(th1)], [p['l1']*np.sin(th1)]])
    q0 = np.array([th1, com2[0, 0], com2[1, 0], th2])
    return q0

  @classmethod
  def simultaneous_impact_velocity(cls, p):
    dth1_val = -1
    dth2_val = 1
    t = sympy.symbols('t')
    th1, th2 = sympy.physics.mechanics.dynamicsymbols('th[1:3]')
    dth1, dth2 = sympy.physics.mechanics.dynamicsymbols('th[1:3]', 1)

    q0 = cls.simultaneous_impact_configuration(p)

    rot = sympy.Matrix([[sympy.cos(th1), -sympy.sin(th1)], [sympy.sin(th1), sympy.cos(th1)]])
    com2 = rot@sympy.Matrix([p['l2']/2*sympy.cos(th2), p['l2']/2*sympy.sin(th2)]) + \
        sympy.Matrix([p['l1']*sympy.cos(th1), p['l1']*sympy.sin(th1)])
    dcom2 = com2.diff(t).subs({th1:q0[0], th2:q0[3], dth1:dth1_val, dth2:dth2_val})

    dq0 = np.array([dth1_val, dcom2[0], dcom2[1], dth2_val]).astype(np.float64)
    return dq0

  @classmethod
  def nominal_parameters(cls):
    p = {'m1':1, 'm2':1, 'l1':.5, 'l2':2/7, 's0':0, 'ks':100}
    p = {'m1':1, 'm2':1, 'l1':.5, 'l2':2/7, 's0':0, 'ks':10}
    p = {'m1':1, 'm2':1, 'l1':.5, 'l2':2/7, 's0':0, 'ks':1}
    p['gamma'] = .3
    return p

  @classmethod
  def draw_config(cls, q, p, draw_a1=True, ax=None):
    pass
