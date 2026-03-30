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
        
##Cycles without mutations##
repet = int(1.E2)
K = 3.E1
d = 10
cells = 1.E5
r=4

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu = np.array([0.,0.])
nu = np.array([0.,0.])

T=1.E1*K/(alpha[0][0]*(K+1))
minrep = 1

#compart_dyn = Gillespie(T, K, repet, cells, 9.E-1, 8, alpha, gamma, mu, nu, 2, 2, d, ratiolen=r, history = False)
#dyn = compart_dyn.evolv(lamcond=True, xcond=True, carryingcaplim=False)
#hist_evol = dyn[0]
#x = dyn[1]
#lam = dyn[2]

#hist_evol = np.array(hist_evol)
#time = [i for i in range(len(hist_evol))]

#compart_dyn_theo = theory_compartA(T, K, d, 9.E-1, 0., 1.E-1, 8, repet, alpha, gamma, mu, nu, ratiolen=r)
#hist_evol_theo, hist_evol_numb = compart_dyn_theo.evol()
#time_theo = [i for i in range(len(hist_evol_theo))]

cd = os.getcwd()
#np.save(cd + '/data/hist_evol_nomut'+str(r)+'.npy', hist_evol)
#np.save(cd + '/data/lamda_nomut'+str(r)+'.npy', lam)
#np.save(cd + '/data/x_nomut'+str(r)+'.npy', x)
#np.save(cd + '/data/time_nomut'+str(r)+'.npy', time)
#np.save(cd + '/data/hist_evol_theo_nomut'+str(r)+'.npy', hist_evol_theo)
#np.save(cd + '/data/time_theo_nomut'+str(r)+'.npy', time_theo)

##Theoretical simulation of cycles with very fast mutations##
repet = int(1.E2)
K = 3.E1
d = 10
cells = 1.E5
r=4

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu = np.array([1.E-2,0.])
nu = np.array([1.E-2,0.])

T=1.E1*K/(alpha[0][0]*(K+1))

compart_dyn = GillespieA(T, K, repet, cells, 9.E-1, 8, alpha, gamma, mu, nu, 2, 2, d, ratiolen=r, history = False)
dyn = compart_dyn.evolv(lamcond=True, xcond=True, carryingcaplim=False)
hist_evol = dyn[0]
x = dyn[1]
lam = dyn[2]

hist_evol = np.array(hist_evol)
time = [i for i in range(len(hist_evol))]

compart_dyn_theo = theory_compartA(T, K, d, 9.E-1, 0., 1.E-1, 8, repet, alpha, gamma, mu, nu, ratiolen=r)
hist_evol_theo, hist_evol_numb = compart_dyn_theo.evol()
time_theo = [i for i in range(len(hist_evol_theo))]

cd = os.getcwd()
np.save(cd + '/data/hist_evol_strongmut'+str(r)+'.npy', hist_evol)
np.save(cd + '/data/lamda_strongmut'+str(r)+'.npy', lam)
np.save(cd + '/data/x_strongmut'+str(r)+'.npy', x)
np.save(cd + '/data/time_strongmut'+str(r)+'.npy', time)
np.save(cd + '/data/hist_evol_theo_strongmut'+str(r)+'.npy', hist_evol_theo)
np.save(cd + '/data/time_theo_strongmut'+str(r)+'.npy', time_theo)

##Cycles with Medium mutations##
repet = int(1.E2)
K = 3.E1
d = 10
cells = 1.E5
r=4

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu=np.array([7.E-3,0.])
nu=np.array([7.E-3,0.])

T=1.E1*K/(alpha[0][0]*(K+1))


compart_dyn = GillespieA(T, K, repet, cells, 9.E-1, 8, alpha, gamma, mu, nu, 2, 2, d, ratiolen=r, history = False)
dyn = compart_dyn.evolv(lamcond=True, xcond=True, carryingcaplim=False)
hist_evol = dyn[0]
x = dyn[1]
lam = dyn[2]

hist_evol = np.array(hist_evol)
time = [i for i in range(len(hist_evol))]

compart_dyn_theo = theory_compartA(T, K, d, 9.E-1, 0., 1.E-1, 8, repet, alpha, gamma, mu, nu, ratiolen=r)
hist_evol_theo, hist_evol_numb = compart_dyn_theo.evol()
time_theo = [i for i in range(len(hist_evol_theo))]

cd = os.getcwd()
np.save(cd + '/data/hist_evol_mediummut'+str(r)+'.npy', hist_evol)
np.save(cd + '/data/lamda_mediummut'+str(r)+'.npy', lam)
np.save(cd + '/data/x_mediummut'+str(r)+'.npy', x)
np.save(cd + '/data/time_mediummut'+str(r)+'.npy', time)
np.save(cd + '/data/hist_evol_theo_mediummut'+str(r)+'.npy', hist_evol_theo)
np.save(cd + '/data/time_theo_mediummut'+str(r)+'.npy', time_theo)

##Cycles with weak mutations##
repet = int(1.E2)
K = 3.E1
d = 10
cells = 1.E5
r=4

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu = np.array([2.E-4,0.])
nu = np.array([2.E-4,0.])

T=1.E1*K/(alpha[0][0]*(K+1))


compart_dyn = GillespieA(T, K, repet, cells, 9.E-1, 8, alpha, gamma, mu, nu, 2, 2, d, ratiolen=r, history = False)
dyn = compart_dyn.evolv(lamcond=True, xcond=True, carryingcaplim=False)
hist_evol = dyn[0]
x = dyn[1]
lam = dyn[2]

hist_evol = np.array(hist_evol)
time = [i for i in range(len(hist_evol))]

compart_dyn_theo = theory_compartA(T, K, d, 9.E-1, 0., 1.E-1, 8, repet, alpha, gamma, mu, nu, ratiolen=r)
hist_evol_theo, hist_evol_numb = compart_dyn_theo.evol()
time_theo = [i for i in range(len(hist_evol_theo))]

cd = os.getcwd()
np.save(cd + '/data/hist_evol_weakmut'+str(r)+'.npy', hist_evol)
np.save(cd + '/data/lamda_weakmut'+str(r)+'.npy', lam)
np.save(cd + '/data/x_weakmut'+str(r)+'.npy', x)
np.save(cd + '/data/time_weakmut'+str(r)+'.npy', time)
np.save(cd + '/data/hist_evol_theo_weakmut'+str(r)+'.npy', hist_evol_theo)
np.save(cd + '/data/time_theo_weakmut'+str(r)+'.npy', time_theo)