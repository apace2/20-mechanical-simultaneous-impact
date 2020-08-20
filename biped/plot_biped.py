import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

nofigurefirst = True
try:
    from figurefirst import FigureLayout
    nofigurefirst = False
except ImportError:
    print("figurefirst is not installed, generating simplified figure")

from . import _data_folder, _fig_folder
from .run_biped import perturbation_suffix
from .Biped import Biped, RigidBiped, PCrBiped, DecoupledBiped

font = {'family':'sans-serif', 'size':15}
mpl.rc('font', **font)
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.markersize'] = 20

line_params = {'lw':8, 'ms':12}
pos_color = 'b'
neg_color = 'r'
sim_color = 'purple'

figname = 'comparison'

try:
    filename = _data_folder / ('.RigidBiped'+perturbation_suffix)
    data_discon = np.load(filename)

    filename = _data_folder / ('.PCrBiped'+perturbation_suffix)
    data_pcr = np.load(filename)

    filename = _data_folder / ('.DecoupledBiped'+perturbation_suffix)
    data_c1 = np.load(filename)

except FileNotFoundError:
    print("Not all data files exist.")
    print("Please run `biped\\run_biped.py`")
    sys.exit(0)


fig = plt.figure()
gs = fig.add_gridspec(nrows=2, ncols=3, left=.18, bottom=.15, top=1)

ax1 = fig.add_subplot(gs[1, 0])
th = data_discon['thetas']
ax1.plot(data_discon['thetas'][th < 0], data_discon['dQ'][th < 0, Biped.ith], color=neg_color)
ax1.plot(data_discon['thetas'][th > 0], data_discon['dQ'][th > 0, Biped.ith], color=pos_color)
ax1.plot(data_discon['thetas'][th == 0], data_discon['dQ'][th == 0, Biped.ith], '.',
         color=sim_color)
ax1.set_ylabel(r'$\dot{\theta}(t=t_f)$')

ax = fig.add_subplot(gs[1, 1], sharey=ax1)
th = data_pcr['thetas']
ax.plot(data_pcr['thetas'][th <= 0], data_pcr['dQ'][th <= 0, Biped.ith], color=neg_color)
ax.plot(data_pcr['thetas'][th >= 0], data_pcr['dQ'][th >= 0, Biped.ith], color=pos_color)
ax.plot(data_pcr['thetas'][th == 0], data_pcr['dQ'][th == 0, Biped.ith], '.', color=sim_color)
plt.setp(ax.get_yticklabels(), visible=False)
ax.set_xlabel(r'$\theta(t=0)$')

ax = fig.add_subplot(gs[1, 2], sharey=ax1)
th = data_c1['thetas']
ax.plot(data_c1['thetas'][th <= 0], data_c1['dQ'][th <= 0, Biped.ith], color=neg_color)
ax.plot(data_c1['thetas'][th >= 0], data_c1['dQ'][th >= 0, Biped.ith], color=pos_color)
ax.plot(data_c1['thetas'][th == 0], data_c1['dQ'][th == 0, Biped.ith], '.', color=sim_color)
plt.setp(ax.get_yticklabels(), visible=False)

###################
# draw the diagrams
###################
p = Biped.nominal_parameters()
q0, _, _ = Biped.ic(p, height=.7)

ax1 = fig.add_subplot(gs[0, 0])
RigidBiped.draw_config(q0, p, ax1)
ax1.axis('off')
ax1.set_ylim((-.1, 1))

ax = fig.add_subplot(gs[0, 1], sharey=ax1)
PCrBiped.draw_config(q0, p, ax)
ax.axis('off')
ax.set_ylim((-.1, 1))

ax = fig.add_subplot(gs[0, 2], sharey=ax1)
DecoupledBiped.draw_config(q0, p, ax)
ax.axis('off')
ax.set_ylim((-.1, 1))

plt.ion()
plt.show()

mpl.rcParams.update(mpl.rcParamsDefault)
