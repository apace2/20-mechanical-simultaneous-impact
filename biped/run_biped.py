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

perturbation_suffix = '_perturb.npz'

def sweep_thetas(hds, thetas=None, complete_trjs=False):
  if thetas is None:
    thetas = np.arange(-.3, .31, .01)
  #thetas = np.arange(-.2, .21, .01)
  thetas[np.abs(thetas) < 1e-6] = 0.

  p = hds.nominal_parameters()

  rx = 1e-5
  dt = 1e-3
  dt = 1e-2
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


if __name__ == '__main__':
  '''
  To call from ipython (with the no-saved argument):
    %run -m biped.run_biped -- --no-saved
  -- : stops ipython parsing the remaining flags
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument("--no-saved", help="Regenerate data, regardless if it already exists",
                      action="store_true")
  args = parser.parse_args()

  for sys in [RigidBiped, DecoupledBiped]:
    filename = _data_folder / ('.' + str(sys)+perturbation_suffix)
    if os.path.exists(filename) and not args.no_saved:
      print("Data already exists in file :", filename)
      print("Not regenerating the data")
      continue

    if sys is RigidBiped:
      # narrower range thetas
      thetas = np.arange(-.21, .22, .01)
      thetas, Q, dQ, _ = sweep_thetas(sys, thetas=thetas)
    else:
      thetas, Q, dQ, _ = sweep_thetas(sys)
    np.savez(filename, thetas=thetas, Q=Q, dQ=dQ)

  filename = _data_folder / ('.'+str(PCrBiped)+perturbation_suffix)
  if os.path.exists(filename) and not args.no_saved:
    print("Data already exists in file :", filename)
    print("Not regenerating the data")

  else:
    thetas, Q, dQ, trjsList = sweep_thetas(PCrBiped, complete_trjs=True)
    np.savez(filename, thetas=thetas, Q=Q, dQ=dQ)
    pickle.dump(trjsList, open(_data_folder / '.PCr_trjs.pkl', 'wb'))
