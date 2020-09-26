import numpy as np
import util

from .Biped import DecoupledBiped, PCrBiped, Biped

biped = DecoupledBiped
biped = PCrBiped

p = biped.nominal_parameters()
q0, dq0, J = biped.ic(p, 0)

dt = 1e-3
rx = 1e-7

tstop_check = .5  #stop before impact
t0 = 0
x_nom_0 = np.hstack((q0, dq0))

trjs_check = util.sim(biped, tstop_check, dt, rx, t0, q0, dq0, J, p)
x_nom_final = np.hstack((trjs_check[0]['q'][-1], trjs_check[0]['dq'][-1]))

sim_trans_matrix = util.variational_soln(trjs_check[0], p, biped.DxF)

perturb = 1e-3
trans_matrix = []
for i in np.identity(len(x_nom_0)):
    x0 = x_nom_0 + perturb*i
    trjs = util.sim(biped, tstop_check, dt, rx, t0, x0[:biped.N_States], x0[biped.N_States:], J, p)
    assert len(trjs) == 1
    x_perturb_final = np.hstack((trjs[0]['q'][-1], trjs[0]['dq'][-1]))
    trans_matrix.append((x_perturb_final - x_nom_final)/perturb)

num_trans_matrix = np.asarray(trans_matrix).T

np.set_printoptions(linewidth=200)
print("Numerical approximation: ")
print(num_trans_matrix)
print("Variational equation aprox: ")
print(sim_trans_matrix)
