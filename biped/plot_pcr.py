import pickle
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import util

nofigurefirst = True
#try:
#    from figurefirst import FigureLayout
#    nofigurefirst = False
#except ImportError:
#    print("figurefirst is not installed, generating simplified figure")

from . import _data_folder, _fig_folder
from .run_biped import perturbation_suffix
from .Biped import RigidBiped, PCrBiped, DecoupledBiped

figname = 'pcr_flow'
foot_style = {'marker':'o', 'markersize':10}
spring_leg_style = {'lw':3, 'color':'grey'}
body_style = {'linestyle':'-', 'lw':10, 'color':'black'}

# plot parameters
font = {'family':'sans-serif', 'size':12}
mpl.rc('font', **font)
xlim_plot = (-.3, .3)
ylim_plot = (-.3, .3)
ylim_plot = (0, .6)
lwl = 8
lwn = 4
ms = 12

# a colormap: [0,1] -> (R,G,B,A), with rgba \in [0,1]
# sort of ...
c_trj_r = 'green'
c_trj_l = 'tomato'
rgb_r = mcolor.to_rgb(mcolor.get_named_colors_mapping()[c_trj_r])
rgb_l = mcolor.to_rgb(mcolor.get_named_colors_mapping()[c_trj_l])
cdict_r = {'red': [(0., rgb_r[0], rgb_r[0]),
                   (1., 0., 0.)],
           'green': [(0., rgb_r[1], rgb_r[1]),
                     (1., 0., 0.)],
           'blue': [(0., rgb_r[2], rgb_r[2]),
                    (1., 0., 0.)]}
cmap_r = mcolor.LinearSegmentedColormap('right', segmentdata=cdict_r, N=256)
cdict_l = {'red': [(0., rgb_l[0], rgb_l[0]),
                   (1., 0., 0.)],
           'green': [(0., rgb_l[1], rgb_l[1]),
                     (1., 0., 0.)],
           'blue': [(0., rgb_l[2], rgb_l[2]),
                    (1., 0., 0.)]}
cmap_l = mcolor.LinearSegmentedColormap('left', segmentdata=cdict_l, N=256)

newcolors = np.vstack((cmap_r(np.linspace(0, 1, 256)),
                       cmap_l(np.linspace(1, 0, 256))))
cmap_tot = mcolor.ListedColormap(newcolors, name='tot')

def color_scale(val, scale=.5):
    # val \in [0,1]
    #sigmoid scaling
    tmp = val*20-10
    return 1/(1+np.exp(-scale*tmp))


def draw_flow(ax, indices, tvals, t_offset=0):
    '''
    ax - axis to draw plot
    indices - indices of theta trajectories to draw
    tvals - times of the internal flow slices
    t_offset - offset of the time labels used

    '''
    c_bgd_simul = 'purple'
    bgd_r = 'red'
    bgd_l = 'blue'

    ind_l = -5
    ind_r = 5
    ms = 10
    lw_regions = 1
    lw_guard = 2
    lw_trj = 1.5

    #tvals = [.9, 1.1]
    #tvals = np.array([.75, 1.1]) + t_offset

    theta_ind = 6


    def colorline(x, y, ax, cmap='copper', z=None):
        '''
        specialized from
        https://stackoverflow.com/a/36074775
        '''
        if z is None:
            z = np.linspace(0, 1, len(x))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = mcoll.LineCollection(segments, array=z, cmap=cmap)
        ax.add_collection(lc)
        return lc

    def plot_theta_time(ind, color, ax, lw=2, dot=False):
        p = PCrBiped.nominal_parameters()
        t, q, dq, o = util.obs(data[ind], PCrBiped, p)

        t = np.hstack(t)+t_offset
        ax.plot(q[:, theta_ind], t, color=color, lw=lw)
        if dot:
            for tval in tvals:
                ind = np.argmin(np.abs(t-tval))
                ax.plot(q[ind, theta_ind], t[ind], '.', color=color, ms=ms, zorder=20)

    def get_subplot_data(tvalue):
        Q0 = []
        dQ = []
        for dat in data:
            p = PCrBiped.nominal_parameters()
            t, q, dq, o = util.obs(dat, PCrBiped, p)
            #t, x, o, p = relax.stack(dat)
            t = np.hstack(t)+t_offset
            #x = np.vstack(x)

            q0 = q[0, theta_ind]

            ind = np.argmin(np.abs(t-tvalue))
            dq_val = dq[ind, theta_ind]

            Q0.append(q0)
            dQ.append(dq_val)

        return Q0, dQ

    def contact(contact_ind, test_value):
        T = []
        Theta = []
        for dat in data:
            try:
                if np.all(dat[contact_ind]['j'] == test_value):
                    T.append(dat[contact_ind-1]['t'][-1]+t_offset)
                    Theta.append(dat[0]['q'][0, theta_ind])
            except IndexError:
                continue
        return T, Theta

    def contact_2(pre_impact, post_impact):
        T = []
        Theta = []
        for dat in data:
            try:
                if np.all(dat[2]['j'] == post_impact) and \
                   np.all(dat[1]['j'] == pre_impact):

                    T.append(dat[1]['t'][-1]+t_offset)
                    Theta.append(dat[0]['q'][0, theta_ind])
            except IndexError:
                continue
        return np.array(T), np.array(Theta)

    ##########################################
    # main plot
    # Theta plot with hatch marks
    ##########################################
    #main

    #plot example trajectories
    plot_theta_time(ind_l, c_trj_l, ax)
    plot_theta_time(ind_r, c_trj_r, ax)

    assert len(data) == 61
    # manually choosing the indices
    mid = 30

    for ind in indices:
        plot_theta_time(ind, cmap_tot(color_scale(ind/len(data))), ax, lw=lw_trj, dot=False)

    ax.set_ylabel('Time $t$ (s)')
    ax.set_xlabel(r'$\theta$ ($t$) (rad)')

    for t in tvals:
        ax.axhline(t, linestyle='--', color='k')

    t_lfirst, theta_lfirst = contact(1, np.array([True, False]))
    t_rfirst, theta_rfirst = contact(1, np.array([False, True]))
    t_sim, theta_sim = contact(1, np.array([True, True]))
    t_l2nd, theta_l2nd = contact_2(np.array([False, True]), np.array([True, True]))
    t_r2nd, theta_r2nd = contact_2(np.array([True, False]), np.array([True, True]))

    # guards
    ax.plot(theta_lfirst, t_lfirst, '-', color=bgd_l, lw=lw_guard, zorder=-1)
    ax.plot(theta_rfirst, t_rfirst, '-', color=bgd_r, lw=lw_guard, zorder=-1)
    ax.plot(theta_l2nd, t_l2nd, '-', color=bgd_l, lw=lw_guard, zorder=-1)
    ax.plot(theta_r2nd, t_r2nd, '-', color=bgd_r, lw=lw_guard, zorder=-1)
    ax.plot(theta_sim, t_sim, '.', color=c_bgd_simul, ms=ms)

    # hatching
    mpl.rcParams['hatch.linewidth'] = lw_regions
    ax.fill_between(theta_lfirst, t_lfirst, t_r2nd, facecolor='none',
                     edgecolor=bgd_l, hatch='\\', linewidth=0, label='-+')
    ax.fill_between(theta_rfirst, t_rfirst, t_l2nd, facecolor='none',
                     edgecolor=bgd_r, hatch='/', linewidth=0, label='+-')
    theta = np.hstack([theta_lfirst[::-1], theta_rfirst[::-1]])
    xvals = np.hstack([t_r2nd[::-1], t_l2nd[::-1]])
    ax.fill_between(theta, xvals, 1.3, facecolor='none', hatch='X', edgecolor=c_bgd_simul,
                     lw=0, zorder=-2, label='++')

    #ystart = -.26
    #yheight = .52
    xstart = -.275
    xwidth = .55

    xstart = -.2
    xwidth = np.abs(xstart)*2+.02
    ylim = [-.1, .2]
    axins_pre = ax.inset_axes([xstart, .75+t_offset, xwidth, .15], transform=ax.transData)
    axins_post = ax.inset_axes([xstart, 1.1+t_offset, xwidth, .15], transform=ax.transData,
                               sharex=axins_pre)

    for t, subax in zip(tvals, [axins_pre, axins_post]):
        q0, dq = get_subplot_data(t)
        #ax.plot(q0, dq)
        zvals = np.vectorize(color_scale)(np.linspace(0, 1, len(q0)))
        colorline(q0, dq, subax, cmap=cmap_tot, z=zvals)
        subax.set_xlim([xstart, xstart+xwidth])
        subax.set_ylim(ylim)
        subax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                          labelleft=False, left=False, right=False)
        subax.plot(q0[ind_l], dq[ind_l], '.', ms=ms, color=c_trj_l)
        subax.plot(q0[ind_r], dq[ind_r], '.', ms=ms, color=c_trj_r)
        subax.text(.01, .12, r'$\dot{\theta}(t='+('%.2f' % t) + ')$', rotation='horizontal')
        subax.text(.05, -.07, r'$\theta(t=0)$', rotation='horizontal')

        ac = 'grey'  #arrow color
        ywidth = ylim[1]-ylim[0]
        ohg = .3
        hw = 2/20
        hl = 1/20

        xhw = hw * ywidth
        xhl = hl * xwidth
        yhw = hw * .15
        yhl = hl * .7
        subax.arrow(xstart, 0, xwidth, 0, fc=ac, ec=ac, lw=1, head_width=xhw, head_length=xhl, ls='-',
                    length_includes_head=True,
                 overhang=ohg)
        subax.arrow(0, ylim[0], 0, ywidth, fc=ac, ec=ac, lw=1, head_width=yhw, head_length=yhl, ls='-',
                    length_includes_head=True, overhang=ohg)

    ax.set_ylim(ylim_plot)
    ax.set_xlim(xlim_plot)
    # flip the y axis
    ax.invert_yaxis()

#draw the diagrams
def draw_fig(ax, ind, tval, p):
    color = cmap_tot(color_scale(ind/len(data)))
    p['body_style']['color'] = color
    t, q, dq, o = util.obs(data[ind], PCrBiped, p)

    tind = np.argmin(np.abs(t-tval))
    q = q[tind]

    PCrBiped.draw_config(q, p, ax)
    ax.axis('off')
    #ax.set_ylim((-.1, 1.2))
    ax.axis('equal')

if __name__ == '__main__':
    try:
        filename_trjs = _data_folder / ('.PCr_trjs.pkl')
        data = pickle.load(filename_trjs.open(mode='rb'))

    except FileNotFoundError:
        print("Not all data files exist.")
        print("Please run `biped\\run_biped.py`")
        sys.exit(0)

    #if nofigurefirst:
    #    plt.rc('text', usetex=True)
    #    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    #    plt.ion()
    #
    #    fig = plt.figure(constrained_layout=True, figsize=(5, 5))
    #    gs1 = fig.add_gridspec(nrows=1, ncols=1)
    #    ax1 = fig.add_subplot(gs1[0, :])
    #    ax1.set_aspect(aspect='equal')
    #else:
    #    layout = FigureLayout(_data_folder/'pcr_outline.svg', make_mplfigures=True)
    #    ax1 = layout.axes['plot']

    #if nofigurefirst:
    #    fig.savefig(_fig_folder / (figname + '.png'))
    #    sys.exit(0)

    fig = plt.figure(figsize=(6, 4))
    #plt.rc('text', usetex=True)
    #lt.rc('text.latex', preamble=r'\usepackage{cmbright}')

    widths = [1, .6, 2, 1]
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=widths, left=.01, right=.99,
                          top=.95, bottom=.15, hspace=0, wspace=0.05)

    layout = {}

    layout['plot'] = fig.add_subplot(gs[:, 2])
    layout['diag_pos_ic'] = fig.add_subplot(gs[0, -1])
    layout['diag_pos_final'] = fig.add_subplot(gs[1, -1])
    layout['diag_neg_ic'] = fig.add_subplot(gs[0, 0])
    layout['diag_neg_final'] = fig.add_subplot(gs[1, 0])

    # flow plot
    ax = layout['plot']
    indices = [10, 17, 24, 30, 36, 43, 50]
    t_offset = -1  # simultaneous impact occurs at t=1 for the nominal configuration
    t_offset = -.7
    tvals = np.array([.75, 1.1]) + t_offset  # values at which to plot cross sections
    draw_flow(ax, indices, tvals, t_offset)
    #ax.set_ylim(ylim_plot)

    p = PCrBiped.nominal_parameters()
    p['foot_style'] = foot_style
    p['spring_leg_style'] = spring_leg_style
    p['body_style'] = body_style

    ax = layout['diag_pos_ic']
    ind = indices[-1]
    draw_fig(ax, ind, tvals[0]-t_offset, p)

    ax = layout['diag_pos_final']
    draw_fig(ax, ind, tvals[1]-t_offset, p)

    ind = indices[0]
    ax = layout['diag_neg_ic']
    draw_fig(ax, ind, tvals[0]-t_offset, p)
    ax = layout['diag_neg_final']
    draw_fig(ax, ind, tvals[1]-t_offset, p)

    filename = _fig_folder / figname
    #layout.save(str(filename)+'.svg')
    #plt.savefig(str(filename) + '.png', dpi=600)
    plt.savefig(str(filename) + '.pdf')
    plt.savefig(str(filename) + '.svg')
    plt.ion()
    plt.show()
