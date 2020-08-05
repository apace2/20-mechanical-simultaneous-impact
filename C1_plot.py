# vim: expandtab tabstop=2 shiftwidth=2

import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt

import util
from DecoupledDP import DecoupledDP as DDP
from dp_c1 import Xi_n1


forwardsim_suffix = 'fowardsim.npz'
varsim_suffix = 'varsim.npz'

sys = DDP

p = sys.nominal_parameters()
p['debug'] = True
J = [0, 0]
dt = 1e-4  # at dt=1e-3 simulatenous impact doesn't occur
dt = 1e-3  #
rx = 1e-7
t0_back = .7

# find initial condition for simultaneous contact
rho = sys.simultaneous_impact_configuration(p)
drho_pre = sys.simultaneous_impact_velocity(p)

#simulate backwards
tstop_back = 0
trjs_back = util.sim(sys, tstop_back, dt, rx, t0_back, rho, drho_pre, J, p)

q0 = trjs_back[-1]['q'][-1]
dq0 = trjs_back[-1]['dq'][-1]
tstop = 2*t0_back
t0 = 0

def forward_sim_perturbation(sys, q0, dq0, p, perturb_dir):
  '''forward simulate a range of perturbations

  sys - system
  q0 - initial configuration
  dq0 - initial velocity
  p - paramters of the system
  perturb_dir- perturbation direcion (len(perturb_q0) = len(q))

  return perturbation, Q, DQ (the final state of the system)
  '''
  assert len(perturb_dir) == sys.DOF

  pert_step = .05
  perturbation = np.arange(-.3, .3+pert_step, pert_step)
  # If a discontinuity appears in plot, check to make sure
  # all trajectories under both impacts

  DQ = []
  Q = []

  # forward simulate perturbations
  for perturb in perturbation:
    q0_tmp = q0 + perturb * perturb_dir
    trjs = util.sim(sys, tstop, dt, rx, t0, q0_tmp, dq0, J, p)
    if len(trjs) != 3:
      print('*************')
      print('WARNING')
      print('Perturbation: '+str(perturb))
      print('Trajectory is of length: '+str(len(trjs)))
      print('Trajectory expected to have 3 segments')
      print('*************')
      print()

    DQ.append(trjs[-1]['dq'][-1])
    Q.append(trjs[-1]['q'][-1])

  DQ = np.vstack(DQ)
  Q = np.vstack(Q)

  return perturbation, Q, DQ

def varsim(sys, q0, dq0):
  '''
  hardcoded for DecoupledDP
  '''

  assert sys == DDP

  dt = 1e-4
  trjs_nom = util.sim(sys, tstop, dt, rx, t0, q0, dq0, J, p)
  X1 = util.variational_soln(trjs_nom[0], p, sys.DxF)
  X2 = util.variational_soln(trjs_nom[-1], p, sys.DxF)
  Xi = np.asarray(Xi_n1).astype(np.float64)
  Phi = X2 @ Xi @ X1
  return Phi


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--no-saved", help="Don't use saved data, if such data exists",
                      action="store_true")
  args = parser.parse_args()

  use_saved_data = True
  decoupled = True

  if args.no_saved:
    use_saved_data = False


  if decoupled:
    filename = '.decoupled-'
    perturb_dir = np.array([1., 0, 0, 0])

  print("Forward simulate perturbations")
  filename_forwardsim = filename+forwardsim_suffix
  if os.path.exists(filename_forwardsim) and use_saved_data:
    with np.load(filename_forwardsim) as data:
      print('>>>>>Using data from '+filename_forwardsim)
      pert_mag = data['perturbation']
      perturb_dir = data['perturbation_dir']
      Q = data['Q']
      DQ = data['DQ']

  else:
    pert_mag, Q, DQ = forward_sim_perturbation(sys, q0, dq0, p, perturb_dir)
    np.savez(filename_forwardsim, perturbation=pert_mag, perturbation_dir=perturb_dir,
             Q=Q, DQ=DQ)

  print('')
  print("Calculate variational solution")
  filename_varsim = filename+varsim_suffix
  if os.path.exists(filename_varsim) and use_saved_data:
    with np.load(filename_varsim) as data:
      print('>>>>>Using data from: '+filename_varsim)
      Phi = data['Phi']

  else:
    Phi = varsim(DDP, q0, dq0)
    np.savez(filename_varsim, Phi=Phi)

  ######
  # Generate plots
  ######
  assert np.all(perturb_dir == np.array([1., 0, 0, 0]))

  pert_vec = np.hstack((perturb_dir, np.zeros_like(perturb_dir)))
  pert_slope = Phi@pert_vec
  zero_perturb_ind = np.isclose(pert_mag, np.zeros_like(pert_mag)).nonzero()[0][0]

  state_ind = 0
  nom_val = Q[zero_perturb_ind, state_ind]

  fig, ax = plt.subplots(1, 1, sharex=True)
  ax.plot(pert_mag, Q[:, state_ind], label='Forward Simulation')  #simulated
  ax.plot(pert_mag, nom_val + pert_slope[state_ind]*pert_mag, label='Linear Approximation')
  ax.set_xlabel('Perturbation magnitude\nPerturbation direction = '+str(perturb_dir))
  ax.set_ylabel('$\Theta_1(t=t_{\text{final}})$')

  ax.set_title('Approximating perturbation of flow with its derivative')
  plt.tight_layout()

  plt.ion()
  plt.show()
