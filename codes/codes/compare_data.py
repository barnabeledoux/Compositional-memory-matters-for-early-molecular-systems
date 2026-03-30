import numpy as np
import matplotlib.pyplot as plt
import random as rand
import scipy
from scipy.optimize import fsolve
import os
from tqdm import tqdm
import sys
from class_comparts import *
from colorsarr import colorsarr
colarr = colorsarr()
carr = colarr.carr
cd = os.getcwd()

class disttodata:
    def __init__(self, data, timedata, theo, timetheo):
        self.data, self.tdata, self.theo, self.ttheo = data, timedata, theo, timetheo
        indcom, i, tdat = [], 0, timedata[0]
        for j, tth in enumerate(timetheo):
            if tth>=tdat and i<len(timedata):
                i+=1
                if i<len(timedata):
                    tdat = timedata[i]
                indcom.append(j)
        self.indcomp = indcom
        self.l = min(len(indcom), len(timedata))
    
    def dist(self):
        norm = 0
        for tind in range(self.l):
            id, it = tind, self.indcomp[tind]
            norm += (self.data[id]-self.theo[it])**2/(((self.data[id]+self.theo[it])/2)**2)
        return np.sqrt(norm/self.l)

    ##Compare with exp##
from scipy.optimize import minimize


d = 5.
cells = 1.E5
repet = int(2.3E1)

alpha = np.array([[2.6,0.],[0.,2.7]])
gamma = np.array([[5.7E0,3.0E0],[0.,3.4E0]])
mut = 5E-5*alpha[0,0]
mu=np.array([mut,0.])
nu=np.array([mut/2, mut/2])
K = 4.1E1
r=4.
s=1.
sarr = np.linspace(0.5, 1., 40)

T=5#1.E1*K/(alpha[0][0]*(K+1))
r1, r2, r3, r4 = 5.100e-01,  1.303e-02,  4.165e-01,  1.077e-01 #.79, 1E-3, 0.48, 0.32

stir, theo, opti = True, True, False

distarr = []

if opti:
    def optifunc(rvec):
        r1, r2, r3, r4 = rvec[0], rvec[1], rvec[2], rvec[3]
        shift2 = 2
        shift = 4
        compart_dyn_theo = theory_compart_bet(T, K, d, 2.2E-1, 3.2E-1, 2.4E-1, 8, repet+int(10), alpha, gamma, mu, nu, ratiolen=r, r1=r1, r2=r2, r3=r3, r4=r4, tradeoff = 4, ncomp = int(cells))
        hist_evol, hist_evol_numb = compart_dyn_theo.evol()
        time_theo = [i for i in range(len(hist_evol))]
        HL1list = [0.218118336,0.227013906,0.241739112,0.029458487,0.013481953,0.011812822,0.006376391,0.004995789,0.001909697,6.94439E-05,1.44793E-05,9.80539E-05,0.002625356,0.126372002,0.422101148,0.247547031,0.032329216,0.009901826,0.004799747,0.002206152,0.000166194,1.32899E-05,2.78631E-06]
        HL2list = [0.32072178, 0.05369015, 0.014985295, 0.00150643, 0.000508845, 0.000225359, 0.000247976, 0.002944985, 0.108122679, 0.109727515, 0.070932843, 0.032033543, 0.022815407, 0.014664688, 0.000851323, 4.64424E-05, 5.84029E-05, 0.001154458, 0.006910299, 0.13068888, 0.232167114, 0.272675814, 0.094491495]
        PL2list = [0.240595437, 0.450775925, 0.173122285, 0.024965157, 0.006356312, 0.004848052, 0.003487809, 0.004154021, 0.29811999, 0.774331997, 0.890651416, 0.954567282, 0.964807628, 0.805719819, 0.087438475, 0.001521278, 0.000206667, 0.000268471, 0.000594932, 0.031763488, 0.102679023, 0.369898367, 0.814597275]
        PL3list = [0.220564447, 0.268520019, 0.570153308, 0.944069926, 0.97965289, 0.983113768, 0.989887824, 0.987905205, 0.591847634, 0.115871045, 0.038401261, 0.013301121, 0.009751609, 0.053243492, 0.489609055, 0.750885248, 0.967405714, 0.988675245, 0.987695022, 0.83534148, 0.664987669, 0.357412529, 0.090908444]
        time_data = range(len(HL1list)-shift2)
        dist = 0
        for tupl in [(time_data, HL1list[shift2:], time_theo[:-shift-shift2], [repart[0][0] for repart in hist_evol[shift+shift2:]]),(time_data, HL2list[shift2:], time_theo[:-shift-shift2], [repart[0][1] for repart in hist_evol[shift+shift2:]]),(time_data, PL2list[shift2:], time_theo[:-shift-shift2], [repart[1][0] for repart in hist_evol[shift+shift2:]]),(time_data, PL3list[shift2:], time_theo[:-shift-shift2], [repart[1][1] for repart in hist_evol[shift+shift2:]])]:
            tdata, data, ttheo, theo = tupl[0], tupl[1], tupl[2], tupl[3]
            disttodat = disttodata(data, tdata, theo, ttheo)
            dist += disttodat.dist()
        return dist

    cons = ({'type': 'ineq', 'fun': lambda x:  x[0]-0.51},
            {'type': 'ineq', 'fun': lambda x:  x[1]},
            {'type': 'ineq', 'fun': lambda x:  x[2]},
            {'type': 'ineq', 'fun': lambda x:  x[3]},
            {'type': 'ineq', 'fun': lambda x:  1.-x[0]},
            {'type': 'ineq', 'fun': lambda x:  .2-x[1]},
            {'type': 'ineq', 'fun': lambda x:  .5-x[2]},
            {'type': 'ineq', 'fun': lambda x:  .5-x[3]},)
    ropti = minimize(optifunc, (.79, 1E-3, 0.18, 0.1), constraints=cons)
    print(ropti)

for n, s in enumerate(tqdm(sarr)):

    if stir:
        compart_dyn_theo = theory_compart_stir_betedge(T, K, d, 2.2E-1, 3.2E-1, 2.4E-1, 8, repet+int(10), alpha, gamma, mu, nu, s= s, ncomp = int(cells), tradeoff=4, ratiolen=r, r1=r1, r2=r2, r3=r3, r4=r4)
        hist_evol_stir = compart_dyn_theo.evol(fraction=True)
        time_stir = [i for i in range(len(hist_evol_stir)+1)]

    #fig1 = plt.figure() 
    #ax1 = fig1.add_subplot(111)

    shift2 = 2
    shift = 4
    #ax1.plot(time_theo[:-shift-shift2],[repart[0][0] for repart in hist_evol[shift+shift2:]],label=r'Replicases $m_0$', c=carr[0], alpha=.5, lw=2.8, ls='-')
    #ax1.plot(time_theo[:-shift-shift2],[repart[0][1] for repart in hist_evol[shift+shift2:]],label=r'Replicases $m_1$', c=carr[2], alpha=.5, lw=2.8, ls='-')
    #ax1.plot(time_theo[:-shift-shift2],[repart[1][0] for repart in hist_evol[shift+shift2:]],label=r'Parasites $y_0$', c=carr[6], alpha=.5, lw=2.8, ls='-')
    #ax1.plot(time_theo[:-shift-shift2],[repart[1][1] for repart in hist_evol[shift+shift2:]],label=r'Parasites $y_1$', c=carr[10], alpha=.5, lw=2.8, ls='-')


    ## DATA ##
    HL1list = [0.218118336,0.227013906,0.241739112,0.029458487,0.013481953,0.011812822,0.006376391,0.004995789,0.001909697,6.94439E-05,1.44793E-05,9.80539E-05,0.002625356,0.126372002,0.422101148,0.247547031,0.032329216,0.009901826,0.004799747,0.002206152,0.000166194,1.32899E-05,2.78631E-06]
    HL2list = [0.32072178, 0.05369015, 0.014985295, 0.00150643, 0.000508845, 0.000225359, 0.000247976, 0.002944985, 0.108122679, 0.109727515, 0.070932843, 0.032033543, 0.022815407, 0.014664688, 0.000851323, 4.64424E-05, 5.84029E-05, 0.001154458, 0.006910299, 0.13068888, 0.232167114, 0.272675814, 0.094491495]
    PL2list = [0.240595437, 0.450775925, 0.173122285, 0.024965157, 0.006356312, 0.004848052, 0.003487809, 0.004154021, 0.29811999, 0.774331997, 0.890651416, 0.954567282, 0.964807628, 0.805719819, 0.087438475, 0.001521278, 0.000206667, 0.000268471, 0.000594932, 0.031763488, 0.102679023, 0.369898367, 0.814597275]
    PL3list = [0.220564447, 0.268520019, 0.570153308, 0.944069926, 0.97965289, 0.983113768, 0.989887824, 0.987905205, 0.591847634, 0.115871045, 0.038401261, 0.013301121, 0.009751609, 0.053243492, 0.489609055, 0.750885248, 0.967405714, 0.988675245, 0.987695022, 0.83534148, 0.664987669, 0.357412529, 0.090908444]
    time_data = range(len(HL1list)-shift2)


    #ax1.scatter(time_data, HL1list[shift2:], color=carr[1], alpha=.5, s=65, marker='o')
    #ax1.scatter(time_data, HL2list[shift2:], color=carr[3], alpha=.5, s=65, marker='o')
    #ax1.scatter(time_data, PL2list[shift2:], color=carr[7], alpha=.5, s=65, marker='o')
    #ax1.scatter(time_data, PL3list[shift2:], color=carr[11], alpha=.5, s=65, marker='o')
    #ax1.plot(time_data, HL1list[shift2:], color=carr[1], alpha=.2, lw=2., ls='--')
    #ax1.plot(time_data, HL2list[shift2:], color=carr[3], alpha=.2, lw=2., ls='--')
    #ax1.plot(time_data, PL2list[shift2:], color=carr[7], alpha=.2, lw=2., ls='--')
    #ax1.plot(time_data, PL3list[shift2:], color=carr[11], alpha=.2, lw=2., ls='--')


    #ax1.set_title(r'Transient compartmentalization dynamics (' +str(repet)+' repetitions)')
    #ax1.set_xlabel(r'Round number')
    #ax1.set_ylabel(r'Fractions')
    #ax1.set_yscale('log')
    #ax1.set_xlim(0, 2.1E1)
    #ax1.set_ylim(2.E-4, 1.8E0)
    #fig1.set_size_inches(15., 5.)

    dist = 0
    for tupl in [(time_data, HL1list[shift2:], time_stir[:-shift-shift2], [repart[0][0] for repart in hist_evol_stir[shift+shift2:]]),(time_data, HL2list[shift2:], time_stir[:-shift-shift2], [repart[0][1] for repart in hist_evol_stir[shift+shift2:]]),(time_data, PL2list[shift2:], time_stir[:-shift-shift2], [repart[1][0] for repart in hist_evol_stir[shift+shift2:]]),(time_data, PL3list[shift2:], time_stir[:-shift-shift2], [repart[1][1] for repart in hist_evol_stir[shift+shift2:]])]:
        tdata, data, ttheo, theo = tupl[0], tupl[1], tupl[2], tupl[3]
        disttodat = disttodata(data, tdata, theo, ttheo)
        dist += disttodat.dist()

    distarr.append(dist)

    if n%5==0:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of subplots
        fig.suptitle(r'$r_1 = {} , $'.format(str(r1)[:4]) + r'$r_2 = {} , $'.format(str(r2)[:4]) + r'$r_3 = {} , $'.format(str(r3)[:4]) + r'$r_4 = {} , $'.format(str(r4)[:4]) + r' and $s = {}$'.format(str(s)[:4]) +r', dist = ${}$ '.format(str(dist)[:5]), fontsize=14)  # Main title for the figure

        if theo:
            compart_dyn_theo = theory_compart_bet(T, K, d, 2.2E-1, 3.2E-1, 2.4E-1, 8, repet+int(10), alpha, gamma, mu, nu, ratiolen=r, r1=r1, r2=r2, r3=r3, r4=r4, tradeoff = 4, ncomp = int(cells))
            hist_evol, hist_evol_numb = compart_dyn_theo.evol()
            time_theo = [i for i in range(len(hist_evol))]

        # List of data and labels
        data_list = [([repart[0][0] for repart in hist_evol[shift+shift2:]], r'$x_0 \,(HL1)$'), ([repart[0][1] for repart in hist_evol[shift+shift2:]], r'$x_1 \,(HL2)$'), ([repart[1][0] for repart in hist_evol[shift+shift2:]], r'$z_0 \,(PL2)$'), ([repart[1][1] for repart in hist_evol[shift+shift2:]], r'$z_1 \,(PL3)$')]
        data_list_stir = [([repart[0][0] for repart in hist_evol_stir[shift+shift2-1:]], r'$x_0 \,(HL1)$'), ([repart[0][1] for repart in hist_evol_stir[shift+shift2-1:]], r'$x_1 \,(HL2)$'), ([repart[1][0] for repart in hist_evol_stir[shift+shift2-1:]], r'$z_0 \,(PL2)$'), ([repart[1][1] for repart in hist_evol_stir[shift+shift2-1:]], r'$z_1 \,(PL3)$')]
        exp_list = [HL1list[shift2:], HL2list[shift2:], PL2list[shift2:], PL3list[shift2:]]
        axes = axs.ravel()  # Flatten the axes array for easy iteration


        cind=[0,2,6,10]
        # Loop through each subplot and plot
        for i, (ax, (data, label)) in enumerate(zip(axes, data_list)):
            data_stir = data_list_stir[i][0]
            if stir:
                ax.plot(time_stir[:-shift-shift2], data_stir, label=label, c=carr[cind[i]], alpha=.5, lw=2.8, ls='-')
            if theo:
                ax.plot(time_theo[:-shift-shift2], data, c=carr[cind[i]], alpha=.3, lw=2.8, ls='--')
            ax.scatter(time_data, exp_list[i], color=carr[cind[i]+1], alpha=.5, s=65, marker='o')
            ax.plot(time_data, exp_list[i], color=carr[cind[i]+1], alpha=.2, lw=2., ls='--')
            ax.set_yscale('log')
            ax.set_xlim(0, 2.1E1)
            ax.set_ylim(1.E-3, 1.E0)
            ax.set_xlabel(r'Round number', fontsize=15, labelpad=-0.3)
            ax.set_ylabel(r'Fractions', fontsize=15)
            ax.set_title(label, fontsize=16)
        fig.set_size_inches(10., 10.)
        plt.savefig('results/gillespie_cycles'+'_frac_theo_'+str(s)[:3]+'.pdf')
        plt.show()

fig2 = plt.figure() 
ax2 = fig2.add_subplot(111)

#distarr = [4.736, 4.664, 4.518, 4.462, 4.377, 4.379, 4.351, 4.374, 4.404, 4.460, 4.442, 4.510, 4.553, 4.639, 4.661, 4.669, 4.670, 4.889, 4.760, 4.955]
np.save(cd + '/data/distance_data.npy', distarr)
ax2.plot(sarr, distarr, c = carr[4], lw=2.3, alpha=0.8, label=r'Distance')
ax2.set_title(r'Distance bewteen experiments and data')
ax2.set_xlabel(r'Stirring $s$')
ax2.set_ylabel(r'Distance')
plt.savefig('results/distance'+'_frac_theo.pdf')

fig2 = plt.figure() 
ax2 = fig2.add_subplot(111)

distarr = np.load(cd + '/data/distance_data.npy')
ax2.plot(sarr, distarr, c = carr[4], lw=2.3, alpha=0.8, label=r'Distance')
ax2.set_title(r'Distance bewteen experiments and data', fontsize=16)
ax2.set_xlabel(r'Stirring $s$', fontsize=15)
ax2.set_ylabel(r'Distance', fontsize=15)
plt.savefig('results/distance'+'_frac_theo_'+str(s)+'.pdf')