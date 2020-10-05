# vim: expandtab tabstop=2 shiftwidth=2

import util
from .Biped import RigidBiped, PCrBiped, DecoupledBiped
from . import _fig_folder

def generate_pertub_anims(hds):
  thetas = [-.1, +0, .1]
  for theta in thetas:
    p = hds.nominal_parameters()
    q0, dq0, J = hds.ic(p, theta)
    J = [False, False]
    rx = 1e-5
    dt = 1e-2
    t0 = 0
    tstop = 1.4
    trjs = util.sim(hds, tstop, dt, rx, t0, q0, dq0, J, p)
    anim = hds.anim(trjs, p, fps=40)
    anim.save(_fig_folder/(str(hds)+str(theta)+'.mp4'))


generate_pertub_anims(RigidBiped)
generate_pertub_anims(PCrBiped)
