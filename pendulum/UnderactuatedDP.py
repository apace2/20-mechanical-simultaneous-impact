# vim: expandtab tabstop=2 shiftwidth=2


import argparse
import numpy as np
import os
import sympy
import sys

from scipy import linalg as la
from scipy import optimize as opt
from tqdm import tqdm

import util
from . import _data_folder
from .DoublePendulum import DoublePendulum


class UnderactuatedDP(DoublePendulum):
  @classmethod
  @util.mechanics_memoization
  def ddq(cls, t, k, q, dq, J, p):
    '''
    Constant applied torque \tau on joint \theta_1
    M ddq + C(q, dq) dq = [\tau, 0]^T

    constant torque is specified in p['tau']
    '''
    M = cls.M(t, k, q, J, p)
    if 'symbolic' in p and p['symbolic']:
      Minv = sympy.Matrix(M).inv()
    else:
      # not the numerically optimal way to solve the equation
      Minv = la.inv(M)

    ddq = Minv@(np.array([p['tau'], 0])-cls.C(t, k, q, dq, J, p)@dq)
    return ddq

  @classmethod
  def DxF(cls, t, k, q, dq, J, p):
    raise NotImplementedError
