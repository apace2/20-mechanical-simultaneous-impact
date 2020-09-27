# vim: expandtab tabstop=2 shiftwidth=2

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm

import biped
import util

from . import _data_folder
from .Biped import RigidBiped, PCrBiped, DecoupledBiped
from .salt_calc import salt_biped

perturbation_suffix = '_perturb.npz'
varsim_suffix = '_var.npz'

def sweep_thetas(hds, thetas=None, complete_trjs=False, dt=1e-2):
  if thetas is None:
    thetas = np.arange(-.3, .31, .01)
  #thetas = np.arange(-.2, .21, .01)
  thetas[np.abs(thetas) < 1e-6] = 0.

  p = hds.nominal_parameters()

  rx = 1e-5
  #dt = 1e-3
  #dt = 1e-2
  t0 = 0
  tstop = 1.4

  Q = []
  dQ = []
  trjsList = []
  for index, theta in enumerate(tqdm(thetas, desc="Simulating "+str(hds)+"...")):
    q0, dq0, J = hds.ic(p, theta)
    J = [False, False]
    trjs = util.sim(hds, tstop, dt, rx, t0, q0, dq0, J, p)
    if len(trjs) != 3 and theta != 0:
      print('*************')
      print('Warning: a trajectory with nonzero perturbation does not have 3 mode transitions')
      print('Number of transitions: '+str(len(trjs)))
      print('Initial theta: '+str(theta))

    Q.append(trjs[-1]['q'][-1])
    dQ.append(trjs[-1]['dq'][-1])
    if complete_trjs:
      trjsList.append(trjs)

  return thetas, Q, dQ, trjsList

def varsim(hds):
  _, _, _, trjsList = sweep_thetas(hds, np.asarray([0.]), True, dt=1e-3)
  trjs_nom = trjsList[0]
  p = hds.nominal_parameters()
  X1 = util.variational_soln(trjs_nom[0], p, hds.DxF)

  X2 = util.variational_soln(trjs_nom[-1], p, hds.DxF)

  rho = trjs_nom[0]['q'][-1]
  drho = trjs_nom[0]['dq'][-1]

  Xi12, Xi21 = salt_biped(rho, drho, hds, p)

  Phi12 = X2 @ Xi12 @ X1
  Phi21 = X2 @ Xi21 @ X1

  return Phi12, Phi21


def main():
  '''
  To call from ipython (with the no-saved argument):
    %run -m biped.run_biped -- --no-saved
  -- : stops ipython parsing the remaining flags
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument("--no-saved", help="Regenerate data, regardless if it already exists",
                      action="store_true")
  parser.add_argument("--fine", help="Run with a step size of dt=.001 as opposed to the default .01",
                      action="store_true")
  args = parser.parse_args()

  if args.fine:
    dt = 1e-3
  else:
    dt = 1e-2

  # calculate perturbations
  for sys in [RigidBiped, DecoupledBiped]:
    filename = _data_folder / ('.' + str(sys)+perturbation_suffix)
    if os.path.exists(filename) and not args.no_saved:
      print("Data already exists in file :", filename)
      print("Not regenerating the data")
      continue

    if sys is RigidBiped:
      # narrower range thetas
      thetas = np.arange(-.21, .22, .01)
      thetas, Q, dQ, _ = sweep_thetas(sys, thetas=thetas, dt=dt)
    else:
      thetas, Q, dQ, _ = sweep_thetas(sys, dt=dt)
    np.savez(filename, thetas=thetas, Q=Q, dQ=dQ)

  filename = _data_folder / ('.'+str(PCrBiped)+perturbation_suffix)
  if os.path.exists(filename) and not args.no_saved:
    print("Data already exists in file :", filename)
    print("Not regenerating the data")

  else:
    thetas, Q, dQ, trjsList = sweep_thetas(PCrBiped, complete_trjs=True, dt=dt)
    np.savez(filename, thetas=thetas, Q=Q, dQ=dQ)
    pickle.dump(trjsList, open(_data_folder / '.PCr_trjs.pkl', 'wb'))

  #calculate variational solutions
  print('\n\n')
  print('Generating variational equation solutions')
  for sys in [PCrBiped, DecoupledBiped]:
    filename = _data_folder / ('.' + str(sys)+varsim_suffix)
    if os.path.exists(filename) and not args.no_saved:
      print("Data already exists in file :", filename)
      print("Not regenerating the data")
      continue

    Phi12, Phi21 = varsim(sys)
    np.savez(filename, Phi12=Phi12, Phi21=Phi21)


if __name__ == '__main__':
  main()
