# vim: expandtab tabstop=2 shiftwidth=2
import copy
import logging
import pickle
from functools import reduce
import numpy as np
import scipy as sp
import scipy.optimize as op

logging.basicConfig(level=logging.INFO)
#logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.WARN)

"""
utility library

functions:
  fp - find fixed point of a func
  jac - numerically approximate Jacobian of a func

Sam Burden, UW Seattle 2017
Andrew Pace, UW Seattle 2020
"""

def linear_interp(t, t0, t1, x0, x1):
  """
  return the linear interpolation value at t1
  given data pair at t0 and t2

  input:
    t1 - desired independent variable
    t0, t2
  """
  return x0 + (x1-x0)/(t1-t0)*(t-t0)

def query_ref_trjs(t, ref_trjs):
  '''
  Given a time t,
  Return from ref_trjs the
    linearly interpolated q, dq
    j and k
  and that time
  '''
  if t < ref_trjs[0]['t'][0]:
    return np.nan, np.nan, np.nan, np.nan
  if t > ref_trjs[-1]['t'][-1]:
    return np.nan, np.nan, np.nan, np.nan

  # select correct trj
  for trj in ref_trjs:
    if t < trj['t'][-1]:
      break

  ind_above = np.nonzero(trj['t'] > t)[0][0]
  q = linear_interp(t, trj['t'][ind_above-1], trj['t'][ind_above],
                    trj['q'][ind_above-1], trj['q'][ind_above])
  dq = linear_interp(t, trj['t'][ind_above-1], trj['t'][ind_above],
                     trj['dq'][ind_above-1], trj['dq'][ind_above])
  return q, dq, trj['j'], trj['k']


def make_trj(t, k, q, dq, J):
  trj = {}
  if isinstance(t, np.ndarray):
    trj['t'] = t
  else:
    trj['t'] = np.hstack(t)
  if isinstance(q, np.ndarray):
    trj['q'] = q
  else:
    trj['q'] = np.vstack(q)
  if isinstance(dq, np.ndarray):
    trj['dq'] = dq
  else:
    trj['dq'] = np.vstack(dq)
  trj['j'] = J
  trj['k'] = k

  return trj


def obs(trjs, sys, params):
  T = []
  Q = []
  dQ = []
  O = {}
  K = []
  J = []

  def appendToKey(key, value):
    try:
      if isinstance(value, dict):
        for subkey in value:
          O[key][subkey].append(value[subkey])
      else:
        O[key].append(value)

    except KeyError:
      if isinstance(value, dict):
        tmp = {}
        for subkey in value:
          tmp[subkey] = [value[subkey]]
        O[_] = tmp
      else:
        O[_] = [o[_]]

  for trj in trjs:
    if isinstance(trj, list):
      continue
    q = trj['q']
    dq = trj['dq']
    t = trj['t']
    k = trj['k']
    j = trj['j']
    index = range(len(t))

    Q.extend(q)
    dQ.extend(dq)
    T.extend(t)
    K.extend([k] * len(index))
    J.extend([j] * len(index))
    #for tmp_q in trj['q']:
      #qext.append(sys.proj(tmp_q))
    for i in index:
      o = sys.O(t[i], k, q[i], dq[i], j, params)

      for _ in o:
        appendToKey(_, o[_])
        #try:
        #  O[_].append(o[_])
        #except KeyError:
        #  O[_] = [o[_]]

  t = np.hstack(T)
  q = np.vstack(Q)
  dq = np.vstack(dQ)
  #qext = np.vstack(qext)
  o = {}
  if O:
    for key in O:
      if isinstance(O[key], dict):
        o[key] = {}
        for subkey in O[key]:
          o[key][subkey] = np.vstack(O[key][subkey])
      else:
        o[key] = np.vstack(O[key])
  else:
    o = None
  return t, q, dq, o

def sim(sys, tstop, dt, rx, t0, q0, dq0, J, params, kstop=None, k0=0):
  '''
  kstop - number of guard transitions to undergo before simulation stopping
  '''
  dt_nominal = dt

  if tstop < t0:
    # simulate in reverse time
    sim_reverse = True
    dt_nominal = -dt_nominal
    dt = -dt
  else:
    sim_reverse = False


  t = t0
  q = q0
  dq = dq0
  k = k0

  d = len(q)

  q_store = [q]
  dq_store = [dq]
  t_store = [t]

  trjs = []
  dt_partial_step = None

  def f(t, k, x, J):
    d = len(x)//2
    q, dq = x[:d], x[d:]
    dx = np.hstack([dq, sys.ddq(t, k, q, dq, J, params)])
    return dx

  while (not sim_reverse and t < tstop) or (sim_reverse and t > tstop):
    if dt_partial_step is not None:
      dt = dt_partial_step
    x = np.hstack([q, dq])

    def wrapper_f(t, x):
      return f(t, k, x, J)
    #dx = util.rk4(partial(f, k=k, J=J), t, x, dt)
    dx = rk4(wrapper_f, t, x, dt)
    x_preview = x + dt*dx

    g = sys.G(t, k, x_preview[:d], x_preview[d:], J, params)

    if np.any(g < -rx):
      dt = dt/2
      logging.info("t: {0:.5e} Step size halving. "
             "dt:{1:4.3e} g:{2}".format(t, dt, g))
      if dt < 1e-10:
        logging.warning('Step size {0:.2} is smaller than 1e-10'.format(dt))
        if 'debug' in params and params['debug']:
          import ipdb
          ipdb.set_trace()
        else:
          import sys
          sys.exit(0)
      continue

    x = x_preview
    q = x[:d]
    dq = x[d:]
    t = t + dt
    if hasattr(sys, 'Halt') and sys.Halt(t, k, q, dq, J, params) and False:
      t_store.append(t)
      q_store.append(q)
      dq_store.append(dq)
      break

    if dt_partial_step is not None:
      dt = dt_nominal
      dt_partial_step = None

    if np.any(g < 0):
      dt = dt_nominal
      t_store.append(t)
      q_store.append(q)
      dq_store.append(dq)

      trj = make_trj(t_store, k, q_store, dq_store, J)
      trjs.append(trj)
      if hasattr(sys, 'Halt') and sys.Halt(t, k, q, dq, J, params) and False:
        return trjs
      if kstop is not None and len(trjs) >= kstop:
        return trjs

      t_store = []
      q_store = []
      dq_store = []

      q, dq_post, J = sys.R(t, k, q, dq, J, params)
      logging.info('t: {0:.5e}: Reset, q: {1}, dq^-: {2}, dq^+:{3}'.format(
        t, q, dq, dq_post))
      dq = dq_post
      k = k + 1

    t_store.append(t)
    q_store.append(q)
    dq_store.append(dq)

  trj = make_trj(t_store, k, q_store, dq_store, J)
  trjs.append(trj)
  return trjs

def rk4(f, t0, x0, dt):
  """
  4th order Runge Kutta
  Inputs:
    f - vector field function with arguments t,x
    t0 - initial time
    x0 - initail state
    dt - step size
  Output:
    dx
  """
  dx1 = f(t0, x0) * dt
  dx2 = f(t0 + 0.5 * dt, x0 + 0.5 * dx1) * dt
  dx3 = f(t0 + 0.5 * dt, x0 + 0.5 * dx2) * dt
  dx4 = f(t0 + dt, x0 + dx3) * dt
  dx = (1.0 / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / dt
  return dx


def Drk4(f, Df, t0, x0, dt):
  """
  Total derivative of the fourth order runge kutta method
  f - vector field function with arguments t,x
  Df - Total spatial derivative of f function with arguments t, x
  t0 - initial time
  x0 - initial state
  dt - step size
  Outputs:
   D(dx)
  """

  v1 = x0
  dx1 = f(t0, x0) * dt
  v2 = x0 + 0.5 * dx1
  dx2 = f(t0 + 0.5 * dt, v2) * dt
  v3 = x0 + 0.5 * dx2
  dx3 = f(t0 + 0.5 * dt, v3) * dt
  v4 = x0 + dx3
  #dx4 = f(t0 + dt, v4)

  n = x0.shape[0]
  I = np.eye(n)
  Ddx1 = Df(t0, v1) * dt
  Ddx2 = Df(t0 + 0.5 * dt, x0 + 0.5 * dx1) @ (I + 0.5 * Ddx1) * dt
  Ddx3 = Df(t0 + 0.5 * dt, v3) @ (I + 0.5 * Ddx2) * dt
  Ddx4 = Df(t0 + dt, v4) @ (I + Ddx3) * dt

  Ddx = 1.0 / 6 * (Ddx1 + 2 * Ddx2 + 2 * Ddx3 + Ddx4) / dt
  return Ddx


def check_Drk4():
  """
  >>> g = 9.81
  >>> def f(t,x):
  ...   return np.array([x[1], -g])
  >>> def Df(t,x):
  ...   return np.array([[ 0., 1. ], [ 0, 0 ]])
  >>> x0 = np.array([ 0, 4. ]); t0 = 0.; dt = .01;
  >>> Ddx = Drk4(f, Df, t0, x0, dt)
  >>> Ddx_numeric = D(lambda x: f(t0, x), x0)
  >>> np.all(np.isclose(Ddx, Ddx_numeric))
  True
  """

  def f(t, x):
    # return np.array([x[1], -g])
    return np.array([x[1] ** 2, -x[0]])

  def Df(t, x):
    # return np.array([[ 0., 1. ], [ 0, 0 ]])
    return np.array([[0.0, 2 * x[1]], [-1, 0]])

  x0 = np.array([0, 4.0])
  t0 = 0.0
  dt = 0.01

  Ddx = Drk4(f, Df, t0, x0, dt)

  Ddx_numeric = D(lambda x: rk4(f, t0, x, dt), x0)
  print(Ddx)
  print("**numeric**")
  print(Ddx_numeric)
  assert np.all(np.isclose(Ddx, Ddx_numeric))


def bisect(f, ab, tol=1e-12, nmax=50):
  """
  Find root of scalar function using bisection
  (search from the right, i.e. from b to a)

  Inputs
    f : R --> R
    ab - (a,b) - interval to search
  (optional)
    tol - terminate when  (b-a)/2 < tol
    nmax - no more than  nmax  bisections

  Outputs
    c - root, i.e.  f(c) ~= 0
  """
  a, b = ab
  for n in range(nmax):
    c = (a + b) / 2.0
    # if ( f(c) == 0 ) or ( (b - a) / 2. < tol ): #Not correct!!
    if f(c) < 0 and f(c) > -tol:
      return b
    if np.sign(f(c)) == np.sign(f(b)):
      b = c
    else:
      a = c
  import ipdb

  ipdb.set_trace()
  return np.nan


def fp(f, x0, eps=1e-6, modes=[1, 2], Ni=4, N=10,
     dist=lambda x, y: np.max(np.abs(x - y))):
  """
  .fp  Find fixed point of f near x0 to within tolerance eps

  The algorithm has two modes:
    1. iterate f; keep result if error decreases initially
    2. run fsolve on f(x)-x; keep result if non-nan

  Inputs:
    f : R^n --> R^n
    x0 - n vector - initial guess for fixed point  f(x0) ~= x0

  Outputs:
    x - n vector - fixed point  f(x) ~= x
  """
  # compute initial error
  xx = f(x0)
  e = dist(xx, x0)
  suc = False
  # 1. iterate f; keep result if error decreases initially
  if 1 in modes:
    # Iterate orbit map several times, compute error
    x = reduce(lambda x, y: f(x), [x0] + range(Ni))
    xx = f(x)
    e = dist(xx, x)
    e0 = dist(x, x0)
    # If converging to fixed point
    if e < e0:
      suc = True
      # Iterate orbit map
      n = 0
      while n < N - Ni and e > eps:
        n = n + 1
        x = xx
        xx = f(x)
        e = dist(xx, x)
      x0 = xx
  # 2. run fsolve on f(x)-x; keep result if non-nan
  if 2 in modes:
    x = x0
    # Try to find fixed point using op.fsolve
    xx = op.fsolve(lambda x: f(x) - x, x)
    # If op.fsolve succeeded
    if not np.isnan(xx).any() or self.dist(xx, x) > e:
      suc = True
      x0 = xx
  # if all methods failed, return nan
  if not suc:
    x0 = np.nan * x0

  return x0


def central(f, x, fx, d):
  """
  df = central()  compute central difference

  df = 0.5*(f(x+d) - f(x-d))/np.linalg.norm(d)
  """
  return 0.5 * (f(x + d) - f(x - d)) / np.linalg.norm(d)


def forward(f, x, fx, d):
  """
  df = forward()  compute forward difference

  df = (f(x+d) - fx)/np.linalg.norm(d)
  """
  return (f(x + d) - fx) / np.linalg.norm(d)


def D(f, x, fx=None, d=1e-6, D=None, diff=forward):
  """
  Numerically approximate derivative of f at x

  Inputs:
    f : R^n --> R^m
    x - n vector
    d - scalar or (1 x n) - displacement in each coordinate
  (optional)
    fx - m vector - f(x)
    D - k x n - directions to differentiate (assumes D.T D invertible)
    diff - func - numerical differencing method

  Outputs:
    Df - m x n - Jacobian of f at x
  """
  if fx is None:
    fx = f(x)
  if D is None:
    D = np.identity(len(x))
  J = [diff(f, x, fx, dd) for dd in list(d * D)]

  return np.array(J).T


def nonsmooth_newton(f, fp, x0, tol=1e-5, maxIter=50, args=()):
  """
  f - function to find zero of
  fp - function that returns PCr of f(x)
  x0 - initial point
  """

  def newton_direction(fx, BF):
    for B in BF:
      d = np.linalg.solve(B[1], -fx)
      if B[0](d):
        return d
    raise ValueError("No directions solved")

  x = x0

  for i in range(maxIter):
    BF = fp(x, *args)
    fx = f(x, *args)
    d = newton_direction(fx, BF)

    x = x + d
    if np.abs(f(x)) < tol:
      return x
  else:
    raise ValueError("Newton Failed to converge")


def dot(*arrays):
  return reduce(np.dot, arrays)

def simple_memoization(func):
  # Very very basic
  func.last_args = None
  func.last_value = None

  def decorator(*args):
    if func.last_args == pickle.dumps(args):
      return func.last_value
    value = func(*args)
    func.last_args = pickle.dumps(args)
    func.last_value = value

    return value

  decorator.__name__ = func.__name__
  decorator.__doc__ = func.__doc__

  return decorator

def mechanics_memoization(func):
  '''
  Very very basic, save most recent call
  assumes function call args are of the form
  func(self, k, t, J, q, dq, p)
  '''
  num_args = 6
  func.last_args = [None]*num_args
  func.last_value = None

  def decorator(*args):
    same_last_arg = False
    for i in [4, 5, 1, 2, 3, 6]:
      if i in [3, 4, 5]:
        if func.last_args[i] is not None and (func.last_args[i][0] != args[i][0] or not np.all(func.last_args[i] == args[i])):
          break
      else:
        if func.last_args[i] != args[i]:
          break
    else:
      same_last_arg = True

    if same_last_arg:
      return func.last_value
    value = func(*args)
    func.last_args = args
    func.last_value = value

    return value

  decorator.__name__ = func.__name__
  decorator.__doc__ = func.__doc__

  return decorator


if __name__ == "__main__":
  import doctest

  doctest.testmod()

  check_Drk4()
