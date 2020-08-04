# vim: expandtab tabstop=2 shiftwidth=2
import os.path
import numpy as np
import matplotlib.pyplot as plt

import util
import var_soln_DP as var_soln
from DoublePendulum import DoublePendulum as DP
from DP_PCr_salt import Xi_n1, Xi_n2

filename_forwardsim = '.dp.forwardsim.npz'
filename_varsim = '.dp.varsim.npz'

p = DP.nominal_parameters()
p['debug'] = True
J = [0, 0]
dt = 1e-4  # at dt=1e-3 simulatenous impact doesn't occur
dt = 1e-3  # still see the PCr nature in perturbation plots
rx = 1e-7
t0_back = .7

# find initial condition for simultaneous contact
q0 = DP.simultaneous_impact_configuration(p)
dq0 = DP.simultaneous_impact_velocity()

#simulate backwards
tstop_back = 0
trjs_back = util.sim(DP, tstop_back, dt, rx, t0_back, q0, dq0, J, p)

q0_forward = trjs_back[-1]['q'][-1]
dq0_forward = trjs_back[-1]['dq'][-1]
tstop = 2*t0_back
t0 = 0

def forward_sim_perturbation(perturb_q0):
  '''forward simulate a range of perturbations

  perturb_q0 - perturb inital q[0] otherwise perturb initial q[1]

  return perturbation, Q, DQ (the final state of the system)
  '''
  pert_step = .05
  perturbation = np.arange(-.3, .3+pert_step, pert_step)
  # If a discontinuity appears in plot, check to make sure
  # all trajectories under both impacts

  DQ = []
  Q = []

  # forward simulate perturbations
  for perturb in perturbation:
    if perturb_q0:
      q0_tmp = q0_forward + np.array([perturb, 0])
    else:
      q0_tmp = q0_forward + np.array([0, perturb])
    trjs = util.sim(DP, tstop, dt, rx, t0, q0_tmp, dq0_forward, J, p)

    #dq_final_store.append(trjs[-1]['dq'][-1])
    DQ.append(trjs[-1]['dq'][-1])
    Q.append(trjs[-1]['q'][-1])
  #dq_final_store = np.vstack(dq_final_store)
  DQ = np.vstack(DQ)
  Q = np.vstack(Q)

  return perturbation, Q, DQ

def varsim():
  dt = 1e-4  # simultaneous impact occurs in simulation
  trjs_nom = util.sim(DP, tstop, dt, rx, t0, q0_forward, dq0_forward, J, p)
  DxF = var_soln.gen_DxF()
  assert np.all(np.isclose(trjs_nom[0]['q'][-1], np.array([0, np.arccos(-2/3)]), atol=1e-7)), \
      "Need to update saltation matrices for different impact configuration"
  assert np.all(np.isclose(trjs_nom[0]['dq'][-1], np.array([-1, 1]))), \
      "Need to update saltation matrices for different impact velocity"

  #first impact guard 0 then 1
  #salt_1_0 = np.array([[-.3, 0, 0, 0],
  #                     [0, -.3, 0, 0],
  #                     [-0.198761597999981, -0.258390077399976, -.3, 0],
  #                     [-0.130809976683737, -0.566843232296196, 0, -.3]])
  #salt_0_1 = np.array([[-.3, 0, 0, 0],
  #                     [0, -.3, 0, 0],
  #                     [-0.0178885438199983, -0.0775170232199927, -.3, 0],
  #                     [-1.45344418537486, -1.88947744098732, 0, -.3]])

  X1 = var_soln.variational_soln(trjs_nom[0], DxF)
  X2 = var_soln.variational_soln(trjs_nom[1], DxF)

  Xin1 = np.asarray(Xi_n1).astype(np.float64)
  Xin2 = np.asarray(Xi_n2).astype(np.float64)

  Phi_10 = X2 @ Xin1 @ X1
  Phi_01 = X2 @ Xin2 @ X1

  return Phi_10, Phi_01


if __name__ == '__main__':
  perturb_q0 = True
  use_saved_data = True

  # forward sim
  print("Forward simulate perturbations")
  if os.path.exists(filename_forwardsim) and use_saved_data:
    with np.load(filename_forwardsim) as data:
      print('>>>>>Using data from '+filename_forwardsim)
      perturbation = data['perturbation']
      Q = data['Q']
      DQ = data['DQ']

  else:
    perturbation, Q, DQ = forward_sim_perturbation(perturb_q0)
    np.savez(filename_forwardsim, perturbation=perturbation, Q=Q, DQ=DQ)

  zero_perturb_ind = np.isclose(perturbation, np.zeros_like(perturbation)).nonzero()[0][0]

  # calcuation variational solution
  print('')
  print("Calculate variational solution")
  if os.path.exists(filename_varsim) and use_saved_data:
    with np.load(filename_varsim) as data:
      print('>>>>>Using data from: '+filename_varsim)
      Phi_10 = data['Phi_10']
      Phi_01 = data['Phi_01']

  else:
    Phi_10, Phi_01 = varsim()
    np.savez(filename_varsim, Phi_01=Phi_01, Phi_10=Phi_10)

  # plot simulated trajectories
  num_subplots = 2
  fig, ax = plt.subplots(num_subplots, 1, sharex=True)
  ax[0].plot(perturbation, Q[:, 0])
  ax[0].set_ylabel('q[0]')
  ax[1].plot(perturbation, Q[:, 1])
  ax[1].set_ylabel('q[1]')

  title_str = "Final state given perturbation of initial "
  if perturb_q0:
    title_str += "q[0]"
  else:
    title_str += "q[1]"
  ax[0].set_title(title_str)

  # plot first order approximation
  if perturb_q0:
    col_ind = 0
  else:
    col_ind = 1
  #position
  ax[0].plot(perturbation, Phi_01[0, col_ind]*perturbation + Q[zero_perturb_ind][0], '--')
  ax[0].plot(perturbation, Phi_10[0, col_ind]*perturbation + Q[zero_perturb_ind][0], '--')

  ax[1].plot(perturbation, Phi_01[1, col_ind]*perturbation + Q[zero_perturb_ind][1], '--')
  ax[1].plot(perturbation, Phi_10[1, col_ind]*perturbation + Q[zero_perturb_ind][1], '--')
  #velocity
  if num_subplots > 2:
    ax[2].plot(perturbation, DQ[:, 0])
    ax[2].set_ylabel('dq[0]')
    ax[3].plot(perturbation, DQ[:, 1])
    ax[3].set_ylabel('dq[1]')
    ax[2].plot(perturbation, Phi_01[2, col_ind]*perturbation + DQ[zero_perturb_ind][0], '--')
    ax[2].plot(perturbation, Phi_10[2, col_ind]*perturbation + DQ[zero_perturb_ind][0], '--')

    ax[3].plot(perturbation, Phi_01[3, col_ind]*perturbation + DQ[zero_perturb_ind][1], '--')
    ax[3].plot(perturbation, Phi_10[3, col_ind]*perturbation + DQ[zero_perturb_ind][1], '--')

  for ax_iter in ax:
    ax_iter.axvline(0, linestyle=':', color='.5')
  fig.tight_layout()
  plt.ion()
  plt.show()

  fig, ax = plt.subplots(1)
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_visible(False)

  blp = {'color':'blue'}
  p['beam_line_param'] = blp
  DP.draw_config(Q[0, :], p, ax=ax)
  DP.draw_config(Q[-1, :], p, ax=ax)
  #blp = {'color':0}
  blp['color'] = '0'
  zero_perturb_ind = int(len(perturbation)/2)
  DP.draw_config(Q[zero_perturb_ind, :], p, ax=ax)

