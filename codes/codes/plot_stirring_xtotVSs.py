import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import random as rand
import scipy
from scipy.optimize import fsolve
import os
from tqdm import tqdm
import pandas as pd
import sys
from colorsarr import colorsarr
colarr = colorsarr()
carr = colarr.carr
cd = os.getcwd()

cmap_name = 'phasecolors'
n_bins = []

twospe = False

def smoothen(list, nm = 10, lim=0):
    reslist = []
    meanlist = [k-nm//2 for k in range(nm)]
    for j in range(len(list)):
        #if abs(list[j] - list[j-1])<0.3:
        if j < len(list) - lim:
            lis = [list[min(max(0,j+k),len(list)-1)] for k in meanlist]
            #if list[j] >= list[j-1] or j<=5:
            reslist.append((sum(lis))/nm)
            #else:
            #    reslist.append(0.)
        else:
            reslist.append(list[j])
        #else:
        #    reslist.append(list[j])
    return reslist

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu = np.array([1.E-2,0.])
nu = np.array([1.E-2,1.E-2])

# Create a single figure with four subplots
xtitle = r'$s$'
ytitle = r'$x_{tot}$'


clist = carr.copy()
clist.reverse()
clist = [(0, clist[2]), (0.1, clist[2]), (0.15, clist[3]), (0.25, clist[4]), (0.42, clist[5]), (0.5, clist[10]),  (0.6, clist[11]), (0.75, clist[12]), (1., clist[13])]
cmapphase = LinearSegmentedColormap.from_list(cmap_name, clist)

list_titles = [r'$K/d$', r'$\alpha T_{mat}$', r'$r$']
if twospe:
    params = [([(8., 1E1, 1., 2.), (1E1, 1E1, 1., 2.), (1.2E1, 1E1, 1., 2.), (1.5E1, 1E1, 1., 2.), (1.8E1, 1E1, 1., 2.)],'K')]
else:
    params = [([(8., 1E1, 1., 2.), (1E1, 1E1, 1., 2.), (1.2E1, 1E1, 1., 2.), (1.5E1, 1E1, 1., 2.), (1.8E1, 1E1, 1., 2.), (2.E1, 1E1, 1., 2.), (2.5E1, 1E1, 1., 2.)],'K'),  #[([(8.E0, 1E1, 1., 10.), (9.E0, 1E1, 1., 10.), (1E1, 1E1, 1., 10.), (1.2E1, 1E1, 1., 10.), (1.5E1, 1E1, 1., 10.), (2.E1, 1E1, 1., 10.), (2.5E1, 1E1, 1., 10.), (3E1, 1E1, 1., 10.), (4E1, 1E1, 1., 10.)],'K'), 
                ([ (2E1, 1E1, 1., .5), (2E1, 1E1, 1., .75), (2E1, 1E1, 1., 1.)],'Tmat'),
                ([ (3E1, 1E1, 1., 10.), (3E1, 1E1, 2., 10.), (3E1, 1E1, 4., 10.), (3E1, 1E1, 5., 10.)],'r')]
list_ind = [0,3,2]
i=0

spread = 1
repet = 3
for param_list, title in params:
    fig1 = plt.figure() 
    ax1 = fig1.add_subplot(111)
    ind = list_ind[i]
    for j, para in enumerate(param_list):
        K, d, r, alphaT = para
        T=alphaT*K/(alpha[0][0]*(K+1))
        if twospe:
            xtotlist = np.load(cd + '/data/xlist_2spe_vs_s'+str(K)+str(d)+str(r)+str(alphaT)+'.npy')
        else:
            xtotlist = np.load(cd + '/data/xlist_vs_s'+str(K)+str(d)+str(r)+str(alphaT)+'.npy')
        for k, x in enumerate(xtotlist):
            if k>=1:
                if x<xtotlist[k-1]:
                    xtotlist[k] = xtotlist[k-1]
        for n in range(repet):
            xtotlist = smoothen(xtotlist, nm=spread, lim = 5 if (i==0 and j==0) else 0)
        slist = np.linspace(0., 1., len(xtotlist))
        if i==0:
            x = (K/d - params[i][0][0][0]/params[i][0][0][1])/(params[i][0][-1][0]/params[i][0][-1][1] - params[i][0][0][0]/params[i][0][0][1])
        else:
            x = (para[list_ind[i]]-params[i][0][0][list_ind[i]])/(params[i][0][-1][list_ind[i]] - params[i][0][0][list_ind[i]])
        ax1.plot(slist, xtotlist, c=cmapphase(x), lw=3.5, alpha=0.7)

    if i == 0:
        norm = mpl.colors.Normalize(vmin=param_list[0][0]/param_list[0][1], vmax=param_list[-1][0]/param_list[-1][1], clip=False)
    else:
        norm = mpl.colors.Normalize(vmin=param_list[0][ind], vmax=param_list[-1][ind], clip=False)
    cbar = fig1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmapphase), ax=ax1, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label(list_titles[i], fontsize=32)
    cbar.ax.tick_params(labelsize=21)

    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    # Loop through each subplot and plot
    ax1.set_xlabel(xtitle, fontsize=32)
    ax1.set_ylabel(ytitle, fontsize=32)
    ax1.tick_params(which='both',direction='in')
    #ax1.legend(loc = 'best', fontsize = 15)
    ax1.set_xlim(0.05, 1.)
    ax1.set_ylim(1.5E-1, 1.)
    ax1.tick_params(axis='both', which='major', labelsize=21)

    fig1.set_size_inches(10., 6.)

    # Adjust layout to prevent overlapping
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the title

    ## Experimental points for K ##
    if i==0:
        datax = np.array(pd.read_excel(cd + '/data/251007_xtot.xlsx'))[5:21, 7:9]
        dataxtot, se = datax[:,0], datax[:,1]

        datas = np.array(pd.read_excel(cd + '/data/251007_xtot.xlsx'))[5:21, 1:2]
        datad = 1/np.array(pd.read_excel(cd + '/data/251007_xtot.xlsx'))[5:21, 2:3]

        datas, datad = np.array([s[0] for s in datas]), np.array([d[0] for d in datad])
        dataind = datad.argsort()
        datas = datas[dataind]
        datad = datad[dataind][::4]
        dataxtot = dataxtot[dataind]
        se = se[dataind]

        marklist = ['o', 's', 'd', 'p']

        for k, d in enumerate(datad):
            sexp, xtotexp, e = datas[4*k:4*(k+1)], dataxtot[4*k:4*(k+1)], se[4*k:4*(k+1)]
            for m, sval in enumerate(sexp):
                val = (0.1 if sval==2. else( 0.2 if sval==5.2 else (0.5 if sval==16. else 0.95)))
                sexp[m] = val
            xind = sexp.argsort()
            s, xtot, e = sexp[xind], xtotexp[xind], e[xind]
            ax1.errorbar(s+(k-1.5)*0.012, xtot, e, c=carr[9+k], lw=0., alpha=0.85, marker=marklist[k], markersize=15, label=r'$d_{\text{exp}}=$'+str(int(d)), capsize=5, capthick=2.5, elinewidth=2.5)
    # Save the combined figure
    if twospe:
        plt.savefig(cd + '/results/xtotVSs_2spe_'+title+'.pdf')
    else:
        ax1.legend(fontsize=21)
        plt.savefig(cd + '/results/xtotVSs_'+title+'.pdf')

    # Indent
    i+=1