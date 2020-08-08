import numpy as np
from tqdm import tqdm

import util
from .Biped import Biped, PCrBiped, DecoupledBiped

def sweep_thetas(hds):
    thetas = np.arange(-.3, .31, .01)
    thetas = np.arange(-.1, .15, .05)
    thetas[np.abs(thetas) < 1e-6] = 0.

    p = hds.nominal_parameters()

    rx = 1e-5
    dt = 1e-3
    t0 = 0
    tstop = 1.3

    Q = []
    dQ = []
    p['debug'] = True
    #for index, theta in enumerate(thetas):
    for index, theta in enumerate(tqdm(thetas, desc="Simulating "+hds.str()+"...")):
        q0, dq0, J = hds.ic(p, theta)
        trjs = util.sim(hds, tstop, dt, rx, t0, q0, dq0, J, p)
        trj = trjs[-1]

        Q.append(trj['q'])
        dQ.append(trj['dq'])

    return thetas, Q, dQ

if __name__ == '__main__':
    thetas, Q, dQ = sweep_thetas(DecoupledBiped)
