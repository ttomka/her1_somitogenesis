import re
import numpy as np
import pylab as plt
from matplotlib import ticker
from pydelay import dde23
from copy import deepcopy 
from scipy.stats.stats import pearsonr
import cPickle as pkl

mycode = """
         double Hn(double y, double y0, double n){
            return 1.0/(1.0 + pow(y/y0, n));}
         """

label = {'p': 'Her1', 'm': 'Her1 mRNA',
         'mH': 'Her mRNA', 'H':  'Her1','mD': 'Dlc mRNA', 'D':  'Dlc', 
         'N':  'Notch', 'mN':  'Notch mRNA', 'I':  'input', 
         'g': 'degradation rate', 
         'gp': '1/2 life pHes [min]', 'ti': 'intronic delay (min)'}

""" plot one frame (concentration in cells of tissue) """
def plot_snapshot(a, clabel=None, clim=None, cbar=False, ax=None, 
                  cmap='gist_ncar', t=None, x=None, o='vertical'):
    if not ax:
        pc = plt.pcolor(a, cmap=cmap)
        plt.gca().set_aspect('equal')
        if clim==None:
            clim = [np.min(a),np.max(a)]
        plt.clim(clim)
        if cbar:
            plt.colorbar(ticks=clim, label=clabel, orientation=o)
        ax = plt.axes()
        pc.cmap.set_under('lightgray')

        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
    else:
        # ax.set(adjustable='box-forced', aspect='equal')
        ax.axes.get_xaxis().set_visible(False)
        if not t:
            ax.axes.get_yaxis().set_visible(False)
        else:
            ax.set_yticklabels(['t=%d'%t])
            ax.set_yticks([x/2.0])
        a = ax.pcolor(a, cmap=cmap)
        a.cmap.set_under('lightgray')
        if clim==None:
            clim = [np.min(a),np.max(a)]
        a.set_clim(clim)
    return

""" plot trajecetories of species concentrations in cells """
def plot_trajectory(ax, pts, species, cells, lw=2, ls='-', legend=True, 
                    log2=True, xlabel=True, a=0.5):
    for x in species:
        for c in cells:
            if not log2:
                ax.plot(pts['t'], pts['%s%s'%(x, c)], alpha=a,
                         label='%s%s'%(label[x],c), lw=lw, ls=ls)
            else:
                ax.plot(pts['t'], np.log2(pts['%s%s'%(x, c)]+1), 
                         label='%s%s'%(label[x],c), lw=lw, ls=ls)
    if xlabel:
        ax.set_xlabel('Time [min]')
    ax.set_ylabel(label[species])
    return

""" series of subplots of frames. for clim find lowest and highest value """
def plot_snap_series(pts, x, y, species, axarr, cbar=False, clim=None,
                     time=True):
    series = []
    for frame in range(len(pts['t'])):
        a, lim = get_matrix(frame, pts, x, y, species)
        series.append(a)
        if clim==None:
            clim = lim
        else:
            clim = [min(clim[0], lim[0]), max(clim[1], lim[1])]
    for frame, t in enumerate(pts['t']):
        if not time:
            t = None
        plot_snapshot(series[frame], species, clim, cbar, axarr[frame], t=t, 
                      x=x)
    plt.xlim([0,y])
    return 

""" sampling dde solution, use pickle to store data (optional)
    return sampled points pts['x'] """
def sample_pts(dde, trange, step, pickle=None):
    pts = dde.sample(trange[0], trange[1], step)
    if pickle!=None:
        pkl.dump(pts, pickle)
    return pts

""" return solution for given parameters;
    specify noisiness of IC and dynamics;
    use spatial gradient in the parameter gradpar """
def run_sim(eqns, x, y, params, conc, noise_factor, gradient, 
            gradpar, bc=None, fix_ic=None, history=None, maxdelay=None):
    noise=None
    if noise_factor!=None:
        noise = {}
        for k in range(1, x*y + 1):
            for s in eqns:
                noise['%s%s'%(s, k)] = '%s*gwn()'%(noise_factor*conc[s][1])
    eqn_arr, ic_arr = write_eqns_and_IC(eqns, x, y, gradient, gradpar, bc, 
                                        fix_ic)
    dde = dde23(eqns=eqn_arr, params=params, noise=noise, supportcode=mycode)
    if not maxdelay:
        maxdelay = max(params['tA'], params['tC'])
    if history==None:
        ic_dic = {'t': np.arange(0.0, max(maxdelay, 5.) + 2*params['dt'], 
                                 params['dt'])}
        for k in eqn_arr.keys():
            ic_dic[k] = ic_arr[k]*np.ones(len(ic_dic['t']))
        dde.hist_from_arrays(ic_dic)
    else:
        dde.hist_from_arrays(history, useend=True)
    dde.set_sim_params(tfinal=params['tf'], dtmax=0.1, dtmin=1e-7, dt0=0.1, 
                       RelTol=1e6, AbsTol=1e6, MaxIter=1e+8)
    dde.run()
    return dde

""" write the array of equations and initial conditions (for each cell);
    for the equations consider the external concentrations given by neighbors
    and possibly spatial gradients in parameter values """
def write_eqns_and_IC(eqns, x, y, gradient, gradpar, bc, fix_ic):
    eqn_arr = {}
    ic_arr = {}
    nb = get_neighbors(x, y, bc)
    k = 1
    for _ in range(x):
        for _ in range(y):
            eqn_arr['p%s'%k] = eqns['p']%(k, k)
            eqn_arr['m%s'%k] = eqns['m']%(k, nb[k]['p'], k)
            if fix_ic==None:
                for s in eqns:
                    ic_arr['%s%s'%(s, k)] = 0.0
            k += 1
    if gradient!=None:
        for key in eqn_arr:
            k = int(re.search(r'\d+$', key).group())
            for p in gradpar:
                eqn_arr[key] = eqn_arr[key].replace(p, 
                                              str(gradient[p][k-1]))
    if fix_ic!=None:
        ic_arr = fix_ic
    return eqn_arr, ic_arr

""" plot (1D) and return (2D) a gradient profile """
def gradient_profile(gradpar, x, y, a, b, front, back, ax=None):
    a = np.concatenate([a*np.ones(front), np.linspace(a, b, y), 
                       b*np.ones(back)])
    grad = np.zeros((x,y+front+back))
    for j in range(x):
        grad[j,:] = a
    if not ax:
        return grad.flatten()
    else: 
        ax.plot(np.arange(y+front+back)+0.5, a, lw=4, c='r')
        ax.set_xticklabels(['P','A'])
        ax.set_xticks([0,y+front+back])
        ax.set_ylabel(gradpar)
        ax.set_yticks([np.min(a), np.max(a)])
        ax.tick_params(labelsize=13)
    return grad.flatten()    



""" compose the sum of relevant neighbor concentrations;
    use bc or zero (None) boundary conditions"""
def get_neighbors(x, y, bc):
    s = 'p'
    delay = 'tC'
    nb = {}
    A = np.zeros((x+2,y+2),dtype=int)
    A[1:(x+1),1:(y+1)] = np.arange(x*y,dtype=int).reshape(x,y) + 1
    if bc in ['cylinder', 'toroid']:
        A[0      ,1:(y+1)] = A[x      ,1:(y+1)]
        A[x+1    ,1:(y+1)] = A[1      ,1:(y+1)]
    if bc=='toroid':
        A[1:(x+1),0      ] = A[1:(x+1),y      ]
        A[1:(x+1),y+1    ] = A[1:(x+1),1      ]
        A[0      ,0      ] = A[x      ,y      ]
        A[0      ,y+1    ] = A[x      ,1      ]
        A[x+1    ,y+1    ] = A[1      ,1      ]
        A[x+1    ,0      ] = A[1      ,y      ]
    k = 1
    for i in range(1,x+1):
        for j in range(1,y+1):
            nb[k] = {}
            nb_nr = [A[i-1,j], A[i,j-1], A[i+1,j], A[i,j+1]] 
            term = '('
            for a in nb_nr:
                if a:
                    term +=  '%s%s(t-%s)+'%(s, a, delay)
            if term.endswith('+'):
                if x*y==2:
                    term = term[:-1] + ')'
                else:    
                    term = term[:-1] + ')/4.0'
            else:
                term += '0)'
            nb[k][s] = term
            k+=1
    return nb

""" arrange datapoints at specific timepoint (frame) as 2D matrix (tissue) """
def get_matrix(frame, pts, x, y, species):
    M = np.zeros((x,y))
    k = 1
    for i in range(x):
        for j in range(y):
            M[i,j] = pts['%s%s'%(species,k)][frame]
            k += 1
    return M, [np.min(M), np.max(M)]

""" monitor the oscillation measures for varied delay values; 
    evaluate in which regions sync_spans the synchronization (or oscillation
    for uncoupled case) condition is satisfied """
def osc_delay_dependence(opar, delay, delay_range, eps_range, eqns, x, y, 
                         params, bc, trange, step, species, eps_spans, 
                         history=None):
    X = {} 
    for eps in eps_range:
        params['e'] = eps
        X[eps] = {'sync': []}
        for op in opar:
            X[eps][op] = {'mean': []}
        for td in delay_range:
            params[delay] = td
            dde = run_sim(eqns, x, y, params, None, None, 
                          None, None, bc, history=history)
            pts = sample_pts(dde, trange, step)
            if x*y>1:
                X[eps]['sync'].append(get_avg_pearsonr(pts, x, y, species))
            for op in opar:
                vals = []
                for c in range(1, x*y + 1):
                    vals.append(osc_param(pts,'%s%s'%(species, c))[op])
                X[eps][op]['mean'].append(np.mean(vals))
        if 'T' in opar:
            # ignore periods smaller than 2
            for i, Ti in enumerate(X[eps]['T']['mean']):
                if Ti<2:
                    X[eps]['T']['mean'][i]= np.nan
    sync_spans = []
    prev_sync = False
    span = [0,0]
    for i, T in enumerate(X[eps_spans]['T']['mean']):
        tau = delay_range[i]
        now_sync = (tau%T)<=T/2. and tau>0 and T>1
        if now_sync and not prev_sync:
            span[0] = tau
        if prev_sync and not now_sync:
            span[1] = delay_range[i-1]
            sync_spans.append(deepcopy(span))    
        prev_sync = now_sync
    if prev_sync:
        span[1]=delay_range[-1]
        sync_spans.append(deepcopy(span))
    return X, sync_spans

""" divide most posterior column of cells """
def tail_growth(history, x, y, eqns):
    hist_new = {}
    for k in range(x*y):
        for s in eqns:
            plus = k/y +1
            hist_new['%s%s'%(s, k+1+plus)] = deepcopy(history['%s%s'%(s, 
                                                                      k+1)])
    hist_new['t'] = deepcopy(history['t'])
    for k in range(x):
        for s in eqns: 
            hist_new['%s%s'%(s, k*(y+1)+1)] = deepcopy(hist_new['%s%s'%(s, 
                                                                  k*(y+1)+2)])
    return hist_new

""" remove most anterior column of cells """
def segmentation(history, x, y, eqns):
    hist_new = {}
    for k in range(x*(y-1)):
        for s in eqns:
            plus = k/(y-1)
            hist_new['%s%s'%(s, k+1)] = deepcopy(history['%s%s'%(s, 
                                                                   k+1+plus)]) 
    hist_new['t'] = deepcopy(history['t'])
    return hist_new

""" record a kymograph of the travelling waves from the previously sampled
    series of tissue snapshots """
def kymograph(times, series, y, cmap, yticks, clim=None, ax=None):
    yticksmid = [x+0.5 for x in yticks]
    G = np.empty((len(times),y))
    G.fill(np.nan)
    for i, M in enumerate(series):
        G[i,:M.shape[1]]=np.mean(M, axis=0)
    if clim==None:
        [np.nanmin(G), np.nanmax(G)]
    print [np.nanmin(G), np.nanmax(G)]
    plot_snapshot(G, None, cmap=cmap, clim=clim, 
                  ax=ax)
    # plt.ylim(0,15)
    # plt.xlabel('Space')
    if ax==None:
        ax= plt.gca()
    # ax.set_ylabel('Time [min]', color='white')
    ax.set_ylim(yticks[0], yticks[-1]+1)
    ax.invert_yaxis()
    # print [x for x in yticks]
    # ax.set_xticklabels(['P','A'])
    # ax.set_xticks([0, y])
    ax.set_yticks(yticksmid)
    ax.set_yticklabels(['$%d$'%times[x] for x in yticks])
    ax.get_yaxis().set_visible(True)
    # ax.get_xaxis().set_visible(True)
    return

""" sensitivity of outvar to variations deltas in parameter par """
def sensitivity(outvar, par, deltas, n, cells, eqns, x, y, params, p, bc,
                conc, trange, step, species, history):
    X = {'d': []}
    for o in outvar:
        X[o] = []
    for d in deltas:
        params[par] = d * p[par]
        dde = run_sim(eqns, x, y, params, conc, None, None, None, bc, 
                      history=history)
        pts = dde.sample(trange[0], trange[1], step)
        X['d'].append(d)
        for o in outvar:
            vals = []
            for c in cells:
                vals.append(osc_param(pts,'%s%s'%(species, c))[o])
            X[o].append(np.mean(vals))
    return X

""" calculate average correlation between all possible pairs of cells in 
    the tissue-history pts"""
def get_avg_pearsonr(pts, x, y, species):
    n = x*y
    p = np.zeros(nchoosek(n, 2))  
    k = 0
    for i in range(1, n+1): 
        for j in range(1,i):
            p[k], _ = pearsonr(pts['%s%s'%(species,i)],
                               pts['%s%s'%(species,j)])
            k += 1
    return np.mean(p)

""" do simulations for a range of Trans-repression delays"""
def heat_map(var, ranges, eqns, x, y, params, bc, fix_ic, conc, trange, step, 
             species, noise=False, cvar='sync', history=None, n=5):
    noise_factor = None
    if cvar=='all':
        X = np.zeros((len(ranges[0])*len(ranges[1]), 5))
    else: 
        X = np.zeros((len(ranges[0])*len(ranges[1]), 3)) 
    i = 0
    for px in ranges[0]:
        for py in ranges[1]:
            # assert physical feasibility
            if var==['tC', 'tA'] and px<py:
                if cvar=='all':
                    X[i,:] = px, py, np.nan, np.nan, np.nan
                else:
                    X[i,:] = px, py, np.nan
                i += 1
                continue
            if noise:
                noise_factor = px
            else:
                params[var[0]] = px
            params[var[1]] = py
            PTS = []
            for _ in range(n):
                dde = run_sim(eqns, x, y, params, conc, noise_factor, None, None, 
                              bc, fix_ic, history)
                PTS.append(dde.sample(trange[0], trange[1], step))
            # assert that there are oscillations
            if 'tA' in var:
                A = []
                for pts in PTS:
                    vals = []
                    for c in range(1, x*y+1):
                        vals.append(osc_param(pts,'%s%s'%(species, c))['amp'])
                    A.append(np.mean(vals))
                if np.mean(A)<5:
                    if cvar=='all':
                        X[i,:] = px, py, np.nan, np.nan, np.nan
                    else:
                        X[i,:] = px, py, np.nan
                    i += 1
                    continue
            vals = []
            for pts in PTS:
                vals.append(get_avg_pearsonr(pts, x, y, species))
            z = np.mean(vals)
            if cvar=='all':
                if z<0.5:
                    z = [np.nan, np.nan, z]
                else:    
                    T = []
                    A = []
                    for pts in PTS:
                        amp = []
                        per = []
                        for c in range(1, x*y+1):
                            per.append(osc_param(pts,'%s%s'%(species, c))['T'])
                            amp.append(osc_param(pts,'%s%s'%(species, c))['amp'])
                        T.append(np.mean(per))
                        A.append(np.mean(amp))
                    z = [np.mean(T), np.mean(A), z]
                X[i,:] = px, py, z[0], z[1], z[2]
            else:
                X[i,:] = px, py, z
            i += 1
    return X

def nchoosek(n, k):
    return np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))

def plot_sync(X, species, ranges, ax_labels, cmap='Blues', s=80, log10=False,
              cvar='sync', cticks=None, fmt=None, mode='Collective'):
    plt.scatter(X[:,0], X[:,1], c=X[:,2], marker='o', cmap=cmap, s=s,
                edgecolors='lightgray', lw=0.5)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    if log10:
        plt.xscale('log', basex=10)
        plt.xlim(10**(1.03*np.log10(ranges[0][0])),
                 10**(0.95*np.log10(ranges[0][-1])))
    else:
        plt.xlim(ranges[0][0]-0.05*ranges[0][0], 1.05*ranges[0][-1])
    plt.ylim(ranges[1][0]-0.05*ranges[1][0], 1.05*ranges[1][-1])
    cb = None
    if cvar=='sync':
        cb = plt.colorbar(label='Synchronization', ticks=[-1, 0, 1])
        plt.clim(vmax=1)
    if cvar=='T':
        cb = plt.colorbar(label='%s period [min]'%mode, ticks=cticks)
    if cvar=='amp':
        cb = plt.colorbar(label='%s amplitude'%mode, ticks=cticks)
    if fmt=='sci':
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.ax.yaxis.offsetText.set_position((5.2,0.5))
    return

""" return oscillation parameters of datapoints s for species v """
def osc_param(s, v, warning=False):
    i_max = (np.r_[False, s[v][1:] > s[v][:-1]] &
             np.r_[s[v][:-1] > s[v][1:], False])
    i_min = (np.r_[False, s[v][1:] < s[v][:-1]] & 
             np.r_[s[v][:-1] < s[v][1:], False])
    if (sum(i_min==True) > 1) & (sum(i_max==True) > 1):
        T_max = s['t'][i_max][-1] - s['t'][i_max][-2]
        t_max = s['t'][i_max][-1]
        T_min = s['t'][i_min][-1] - s['t'][i_min][-2]
        v_mean = (np.mean(s[v][(s['t'] > s['t'][i_max][-2]) & 
                  (s['t'] < s['t'][i_max][-1])]))
        v_max = s[v][i_max][-1]
        v_min = s[v][i_min][-1]        
    else:
        T_max = 0.0
        T_min = 0.0
        v_mean= np.mean(s[v])
        v_max = v_mean
        v_min = v_mean
        t_max = 0
    if warning:
        if T_max > 0.1:
            if abs(1.0 - T_min/T_max) > 0.1:
                print ('Warning: (T_max-T_min)/T_max higher than 10%', 
                    T_min, T_max)
                T_max = T_min = 0.0
    return {'T'    : 0.5*(T_min + T_max), 
            'amp'  : v_max - v_min, 
            'max'  : v_max, 
            'min'  : v_min,
            'mean' : v_mean,
            't_max': t_max}
