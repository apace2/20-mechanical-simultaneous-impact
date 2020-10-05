# vim: expandtab tabstop=2 shiftwidth=2
import copy
import numpy as np

from pendulum import _fig_folder, _data_folder
import util
from .DoublePendulum import DoublePendulum as DP
from .UnderactuatedDP import UnderactuatedDP

filename = _data_folder / '.nonsmooth_control.npz'


def ic():
  p = DP.nominal_parameters()
  J = [0, 0]
  dt = 1e-3  # still see the PCr nature in perturbation plots
  rx = 1e-7
  t0_back = .7

  # find initial condition for simultaneous contact
  q0 = DP.simultaneous_impact_configuration(p)
  dq0 = DP.simultaneous_impact_velocity()

  #simulate backwards
  tstop_back = 0
  trjs_back = util.sim(DP, tstop_back, dt, rx, t0_back, q0, dq0, J, p)

  q0 = trjs_back[-1]['q'][-1]
  dq0 = trjs_back[-1]['dq'][-1]

  return q0, dq0

def animate():
  q0_nom, dq0 = ic()

  p = DP.nominal_parameters()
  J = [0, 0]
  dt = 1e-3  # still see the PCr nature in perturbation plots
  rx = 1e-7
  tstop = 1.4
  tstart = 0

  for perturb in [-.1, 0, .1]:
    q0 = q0_nom+np.array([perturb, 0.])
    trjs = util.sim(DP, tstop, dt, rx, tstart, q0, dq0, J, p)
    ani = DP.anim(trjs, p)
    ani.save(_fig_folder/('DP'+str(perturb)+'.mp4'))

def animate_control():
  try:
    data = np.load(filename)
  except FileNotFoundError:
    print("Run pendulum/nonsmooth_controller.py before generating animations.")

  p = DP.nominal_parameters()
  J = [0, 0]
  dt = 1e-3  # still see the PCr nature in perturbation plots
  rx = 1e-7
  tstop = 1.4
  tstart = 0
  #pert_dir = data['perturbation_dir']

  pert_mag = data['perturbation']
  tau = data['tau']

  q0_nom, dq0 = ic()
  for perturb in [-.05, 0, .05]:
    ind = np.where(np.isclose(pert_mag, perturb))[0][0]
    p_run = p.copy()
    p_run['tau'] = tau[ind]
    q0 = q0_nom+np.array([perturb, 0.])
    trjs = util.sim(UnderactuatedDP, tstop, dt, rx, tstart, q0, dq0, J, p_run)
    ani = DP.anim(trjs, p)
    ani.save(_fig_folder/('DP_control'+str(perturb)+'.mp4'))

animate()
animate_control()
