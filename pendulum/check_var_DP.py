# vim: expandtab tabstop=2 shiftwidth=2

import sympy
import numpy as np
import util
from DoublePendulum import DoublePendulum

def check_variational_soln():
  dp = DoublePendulum()
  p = dp.nominal_parameters()
  q0 = dp.simultaneous_impact_configuration(p)
  dq0 = dp.simultaneous_impact_velocity()
  J = [0, 0]
  dt = 1e-4  #at dt=1e-3 simulatenous impact doesn't occur
  rx = 1e-7
  t0_back = .7
  tstop_back = 0
  trjs_back = util.sim(dp, tstop_back, dt, rx, t0_back, q0, dq0, J, p)
  q0_forward = trjs_back[-1]['q'][-1]
  dq0_forward = trjs_back[-1]['dq'][-1]
  t0 = 0

  print('Checking variational solution')
  tstop_check = .5
  dt_check = 1e-3
  trjs_check = util.sim(dp, tstop_check, dt_check, rx, t0, q0_forward, dq0_forward, J, p)
  assert len(trjs_check) == 1
  q_final = trjs_check[0]['q'][-1]
  dq_final = trjs_check[0]['dq'][-1]
  x_nom_0 = np.hstack((q0_forward, dq0_forward))
  x_nom_final = np.hstack((q_final, dq_final))
  #
  perturb = 1e-3
  trans_matrix = []
  for i in np.identity(4):
      print('perturbation: '+str(perturb*i))
      x0 = x_nom_0 + perturb*i
      trjs = util.sim(dp, tstop_check, dt_check, rx, t0, x0[:2], x0[2:], J, p)
      assert len(trjs) == 1
      x_perturb_final = np.hstack((trjs[0]['q'][-1], trjs[0]['dq'][-1]))
      trans_matrix.append((x_perturb_final - x_nom_final)/perturb)

  num_trans_matrix = np.asarray(trans_matrix).T
  sim_trans_matrix = util.variational_soln(trjs_check[0], p, dp.DxF)

  print("Numerical approximation: ")
  print(num_trans_matrix)
  print("Variational equation aprox: ")
  print(sim_trans_matrix)


if __name__ == '__main__':
  check_variational_soln()
