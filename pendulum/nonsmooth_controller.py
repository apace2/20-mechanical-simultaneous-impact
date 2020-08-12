# vim: expandtab tabstop=2 shiftwidth=2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy import optimize as opt
from tqdm import tqdm

from .UnderactuatedDP import UnderactuatedDP
from .DoublePendulum import DoublePendulum as DP
from . import _data_folder
import util

filename = _data_folder / '.nonsmooth_control.npz'

def generate_nonsmooth_controller():
  # nominal trajectory and desired state without any applied torque and undergoes
  #  simultaneous impact

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

  q0 = trjs_back[-1]['q'][-1]
  dq0 = trjs_back[-1]['dq'][-1]
  tstop = 2*t0_back
  t0 = 0

  trjs = util.sim(DP, tstop, dt, rx, t0, q0, dq0, J, p)

  desired_q1 = trjs[-1]['q'][-1][1]
  pert_dir = np.array([1, 0])

  #####
  # generate controller
  # have scipy.opt find root that is the torque given an initial perturbation in the state

  # possible methods (all don't require derivaties) for scipy.optimize.root_scalar
  # bisect, toms748
  # bracket is another method, though not sure if a opt.root_scalar method

  def final_q1_given_tau(tau, q0_pert):
    '''
    return \theta[1](t=tstop) when a constant torque tau is applied to theta[0]
    '''
    p_run = p.copy()
    p_run['tau'] = tau
    trjs = util.sim(UnderactuatedDP, tstop, dt, rx, t0, q0+q0_pert, dq0, J, p_run)
    return trjs[-1]['q'][-1][1] - desired_q1


  Tau = []
  pert_mag = np.arange(-.2, .2, .01)
  for mag in tqdm(pert_mag, desc="Finding controller..."):
    perturbation = pert_dir*mag

    # bound the input, otherwise the simulation does not finish
    if mag < -.1:
      bracket = [-.1, 1]

    elif mag >= -.1 and mag < .09:
      bracket = [-.1, .1]

    elif mag >= .09 and mag < .35:
      bracket = [-.5, 1]

    elif mag >= .35 and mag < .43:
      bracket = [-1., 1]

    soln = opt.root_scalar(final_q1_given_tau, method='bisect',
                           bracket=bracket, x0=0, args=(perturbation,))
    Tau.append(soln.root)

  Tau = np.array(Tau)

  # find the resulting final total state
  print("Calculate the final state given the constant torque input")
  Q = []
  DQ = []
  Q_NoControl = []
  DQ_NoControl = []

  for tau, mag in tqdm(zip(Tau, pert_mag), desc="Calculating final state..."):
    q0_pert = pert_dir * mag
    p_control = p.copy()
    p_control['tau'] = tau
    trjs = util.sim(UnderactuatedDP, tstop, dt, rx, t0, q0+q0_pert, dq0, J, p_control)

    Q.append(trjs[-1]['q'][-1])
    DQ.append(trjs[-1]['dq'][-1])

    # Resulting uncontrolled state
    trjs = util.sim(UnderactuatedDP, tstop, dt, rx, t0, q0+q0_pert, dq0, J, p_control)
    Q_NoControl.append(trjs[-1]['q'][-1])
    DQ_NoControl.append(trjs[-1]['dq'][-1])

  np.savez(filename, perturbation=pert_mag, perturbation_dir=pert_dir, tau=Tau,
           QFinal=Q, DQFinal=DQ, Q_NoControlFinal=Q_NoControl, DQFinal_NoControl=DQ_NoControl)

def plot_nonsmooth_controller():
  data = np.load(filename)
  pert_dir = data['perturbation_dir']
  pert_mag = data['perturbation']
  tau = data['tau']


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--no-saved", help="Don't use saved data, if such data exists",
                      action="store_true")
  args = parser.parse_args()

  use_saved_data = True

  if args.no_saved:
    use_saved_data = False

  if os.path.exists(filename) and use_saved_data:
    print('Using nonsmooth controller data from file: '+str(filename))
    print("To regenerate data, delete data file or use flag '--no-saved'")

  else:
    generate_nonsmooth_controller()

  plot_nonsmooth_controller()

