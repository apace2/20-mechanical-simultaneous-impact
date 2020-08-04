# vim: expandtab tabstop=2 shiftwidth=2

import sympy
import numpy as np
import util
from DoublePendulum import DoublePendulum

def variational_soln(trj, p, DxF):
  '''
  compute the variational solution along a trajectory for a mechanical system
  '''
  X0 = np.identity(2*len(trj['q'][0]))

  X = X0
  k = trj['k']
  J = trj['j']
  t = trj['t'][0]

  for tnext, q, dq in zip(trj['t'][1:], trj['q'], trj['dq']):
    dt = tnext - t
    # calculate DxF at the current point
    X = X + dt*DxF(t, k, q, dq, J, p)@X
    t = tnext

  return X
