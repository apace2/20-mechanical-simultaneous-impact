# vim: expandtab tabstop=2 shiftwidth=2
import copy
from scipy import linalg as la
import numpy as np

class fancytype(type):
  # from https://stackoverflow.com/questions/8144026/how-to-define-a-str-method-for-a-class
  def __str__(cls):
    return cls.__name__

class HybridSystem(metaclass=fancytype):
  N_States = None  #number of configuration states

  @classmethod
  def Delta(cls, t, k, q, dq, J, p):
    d = len(q)
    gamma = p['gamma']
    M = cls.M(t, k, q, J, p)
    Da = cls.DaTot(t, k, q, J, p)
    Lambda = cls.Lambda(t, k, q, J, p)

    Delta = np.identity(d) - (1+gamma)*la.inv(M)@Da.T@Lambda@Da
    return Delta

  @classmethod
  def G(cls, t, k, q, dq, J, p):
    a = cls.a(t, k, q, J, p)
    lamb = cls.lamb(t, k, q, dq, J, p)  #constraint force for unilat and bilateral
    g = np.zeros_like(J) * np.nan
    g[np.logical_not(J)] = a[np.logical_not(J)]
    #g[J] = lamb[:np.sum(J)]
    for lambind, gind in enumerate(np.where(J)[0]):
      g[gind] = lamb[lambind]

    return g

  @classmethod
  def O(cls, t, k, q, dq, J, p):
    o = {}
    o['q'] = q
    o['dq'] = dq
    o['g'] = cls.G(t, k, q, dq, J, p)
    return o

  @classmethod
  def R(cls, t, k, q, dq, J, p):
    a = cls.a(t, k, q, J, p)
    Jimpact = a < 0
    Jnew = np.logical_or(J, Jimpact)
    Delta = cls.Delta(t, k, q, dq, Jnew, p)
    dq_ = Delta@dq
    if p['gamma'] > 0:  #elastic collision
      return q, dq_, copy.copy(J)

    if p['gamma'] == 0:  #plastic collision
      return q, dq_, copy.copy(Jnew)

  @classmethod
  def DaTot(cls, t, k, q, J, p):
    Da = np.vstack((cls.Da(t, k, q, J, p),
                    cls.Db(t, k, q, J, p)))
    return Da

  @classmethod
  def DtDaTot(cls, t, k, q, dq, J, p):
    DtDa = np.vstack((cls.DtDa(t, k, q, dq, J, p),
                      cls.DtDb(t, k, q, dq, J, p)))
    return DtDa

  @classmethod
  def Lambda(cls, t, k, q, J, p):
    Da = cls.DaTot(t, k, q, J, p)
    Minv = la.inv(cls.M(t, k, q, J, p))

    Mtmp = Da@Minv@Da.T
    if len(Mtmp) > 0:
      Lambda = la.inv(Mtmp)
    else:
      Lambda = np.array([])
    return Lambda

  @classmethod
  def lamb(cls, t, k, q, dq, J, p):
    Da = cls.DaTot(t, k, q, J, p)
    DtDa = cls.DtDaTot(t, k, q, dq, J, p)

    Minv = la.inv(cls.M(t, k, q, J, p))
    Lambda = cls.Lambda(t, k, q, J, p)
    f = cls.f(t, k, q, dq, J, p)
    lam = -Lambda@(Da@Minv@f+DtDa@dq)
    return lam

  @classmethod
  def ddq(cls, t, k, q, dq, J, p):
    Minv = la.inv(cls.M(t, k, q, J, p))
    f = cls.f(t, k, q, dq, J, p)

    Da = cls.DaTot(t, k, q, J, p)
    if len(Da) == 0:  # no active constraints
      ddq = Minv@f
    else:
      lamb = cls.lamb(t, k, q, dq, J, p)
      ddq = Minv@(f + Da.T@lamb)
    assert ddq.shape == (cls.N_States,)
    return ddq

  @classmethod
  def f(cls, t, k, q, dq, J, p):
    raise NotImplementedError

  @classmethod
  def M(cls, t, k, q, J, p):
    raise NotImplementedError

  #  unilateral constraint
  @classmethod
  def a(cls, t, k, q, J, p):
    raise NotImplementedError

  @classmethod
  def Da(cls, t, k, q, J, p):
    raise NotImplementedError

  @classmethod
  def DtDa(cls, t, k, q, dq, J, p):
    raise NotImplementedError

  @classmethod
  def b(cls, t, k, q, J, p):
    raise NotImplementedError

  @classmethod
  def Db(cls, t, k, q, J, p):
    raise NotImplementedError

  @classmethod
  def DtDb(cls, t, k, q, dq, J, p):
    raise NotImplementedError

  @classmethod
  def simultaneous_impact_configuration(cls, p):
    raise NotImplementedError

  @classmethod
  def simultaneous_impact_velocity(cls):
    raise NotImplementedError

  @classmethod
  def nominal_parameters(cls):
    raise NotImplementedError

  @classmethod
  def draw_config(cls, q, p, ax=None):
    raise NotImplementedError
