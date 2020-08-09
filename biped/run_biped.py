# vim: expandtab tabstop=2 shiftwidth=2

import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm

import util
from .Biped import RigidBiped, PCrBiped, DecoupledBiped

perturbation_suffix = '_perturb.npz'

def sweep_thetas(hds, complete_trjs=False):
  thetas = np.arange(-.3, .31, .01)
  thetas[np.abs(thetas) < 1e-6] = 0.

  p = hds.nominal_parameters()

  rx = 1e-5
  dt = 1e-3
  t0 = 0
  tstop = 1.3

  Q = []
  dQ = []
  trjsList = []
  for index, theta in enumerate(tqdm(thetas, desc="Simulating "+str(hds)+"...")):
    q0, dq0, J = hds.ic(p, theta)
    J = [False, False]
    trjs = util.sim(hds, tstop, dt, rx, t0, q0, dq0, J, p)
    trj = trjs[-1]

    Q.append(trj['q'])
    dQ.append(trj['dq'])
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
    filename = '.'+str(sys)+perturbation_suffix
    if os.path.exists(filename) and not args.no_saved:
      print("Data already exists in file :"+filename)
      print("Not regenerating the data")
      continue

    thetas, Q, dQ = sweep_thetas(sys)
    np.savez(filename, thetas=thetas, Q=Q, dQ=dQ)

  filename = '.'+str(PCrBiped)+perturbation_suffix
  if os.path.exists(filename) and not args.no_saved:
    print("Data already exists in file :"+filename)
    print("Not regenerating the data")

  else:
    thetas, Q, dQ, trjsList = sweep_thetas(PCrBiped, complete_trjs=True)
    np.savez(filename, thetas=thetas, Q=Q, dQ=dQ)
    pickle.dump(trjsList, os.open('.PCr_trjs.pkl', 'wb'))
