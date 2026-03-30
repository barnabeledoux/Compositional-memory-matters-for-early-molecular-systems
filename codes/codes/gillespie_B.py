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
r=1

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu = np.array([0.,0.])
nu = np.array([0.,0.])

T=1.E1*K/(alpha[0][0]*(K+1))

compart_dyn = GillespieB(T, K, repet, cells, 9.E-1, 8, alpha, gamma, mu, nu, 2, 2, d, ratiolen=r, history = False)
dyn = compart_dyn.evolv(fraction = True, lamcond=True, xcond=True, carryingcaplim=False)
hist_evol = np.array(dyn[0])
x = dyn[1]
lam = dyn[2]
time = [i for i in range(len(hist_evol))]

compart_dyn_theo = theory_compart(T, K, d, 9.E-1, 0., 1.E-1, 8., repet, alpha, gamma, mu, nu, ratiolen=r, ncomp=int(cells))
hist_evol_theo = compart_dyn_theo.evol()[0]
time_theo = [i for i in range(len(hist_evol_theo))]

cd = os.getcwd()
np.save(cd + '/data/hist_evol_nomut_hosttopara'+str(r)+'.npy', hist_evol)
np.save(cd + '/data/lamda_nomut_hosttopara'+str(r)+'.npy', lam)
np.save(cd + '/data/x_nomut_hosttopara'+str(r)+'.npy', x)
np.save(cd + '/data/time_nomut_hosttopara'+str(r)+'.npy', time)
np.save(cd + '/data/hist_evol_theo_nomut_hosttopara'+str(r)+'.npy', hist_evol_theo)
np.save(cd + '/data/time_theo_nomut_hosttopara'+str(r)+'.npy', time_theo)

##Theoretical simulation of cycles with very fast mutations##

repet = int(1.E2)
K = 3.E1
d = 10
cells = 1.E5
r=1

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu = np.array([1.E-2,0.])
nu = np.array([1.E-2,1.E-2])

T=1.E1*K/(alpha[0][0]*(K+1))

compart_dyn = GillespieB(T, K, repet, cells, 9.E-1, 8, alpha, gamma, mu, nu, 2, 2, d, ratiolen=r, history = False)
dyn = compart_dyn.evolv(fraction = True, lamcond=True, xcond=True, carryingcaplim=False)
hist_evol = np.array(dyn[0])
x = dyn[1]
lam = dyn[2]
time = [i for i in range(len(hist_evol))]

compart_dyn_theo = theory_compart(T, K, d, 9.E-1, 0., 1.E-1, 8., repet, alpha, gamma, mu, nu, ratiolen=r, ncomp=int(cells))
hist_evol_theo = compart_dyn_theo.evol()[0]
time_theo = [i for i in range(len(hist_evol_theo))]

cd = os.getcwd()
np.save(cd + '/data/hist_evol_fastmut_hosttopara'+str(r)+'.npy', hist_evol)
np.save(cd + '/data/lamda_fastmut_hosttopara'+str(r)+'.npy', lam)
np.save(cd + '/data/x_fastmut_hosttopara'+str(r)+'.npy', x)
np.save(cd + '/data/time_fastmut_hosttopara'+str(r)+'.npy', time)
np.save(cd + '/data/hist_evol_theo_fastmut_hosttopara'+str(r)+'.npy', hist_evol_theo)
np.save(cd + '/data/time_theo_fastmut_hosttopara'+str(r)+'.npy', time_theo)

##Theoretical simulation of cycles with medium mutations##

repet = int(1.E2)
K = 3.E1
d = 10
cells = 1.E5
r=1

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu=np.array([7.E-3,0.])
nu=np.array([7.E-3,7.E-3])

T=1.E1*K/(alpha[0][0]*(K+1))

compart_dyn = GillespieB(T, K, repet, cells, 9.E-1, 8, alpha, gamma, mu, nu, 2, 2, d, ratiolen=r, history = False)
dyn = compart_dyn.evolv(fraction = True, lamcond=True, xcond=True, carryingcaplim=False)
hist_evol = np.array(dyn[0])
x = dyn[1]
lam = dyn[2]
time = [i for i in range(len(hist_evol))]

compart_dyn_theo = theory_compart(T, K, d, 9.E-1, 0., 1.E-1, 8., repet, alpha, gamma, mu, nu, ratiolen=r, ncomp=int(cells))
hist_evol_theo = compart_dyn_theo.evol()[0]
time_theo = [i for i in range(len(hist_evol_theo))]

cd = os.getcwd()
np.save(cd + '/data/hist_evol_mediummut_hosttopara'+str(r)+'.npy', hist_evol)
np.save(cd + '/data/lamda_mediummut_hosttopara'+str(r)+'.npy', lam)
np.save(cd + '/data/x_mediummut_hosttopara'+str(r)+'.npy', x)
np.save(cd + '/data/time_mediummut_hosttopara'+str(r)+'.npy', time)
np.save(cd + '/data/hist_evol_theo_mediummut_hosttopara'+str(r)+'.npy', hist_evol_theo)
np.save(cd + '/data/time_theo_mediummut_hosttopara'+str(r)+'.npy', time_theo)

##Theoretical simulation of cycles with slow mutations##

repet = int(1.E2)
K = 3.E1
d = 10
cells = 1.E5
r=1

alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu = np.array([2.E-4,0.])
nu = np.array([2.E-4,2.E-4])

T=1.E1*K/(alpha[0][0]*(K+1))

compart_dyn = GillespieB(T, K, repet, cells, 9.E-1, 8, alpha, gamma, mu, nu, 2, 2, d, ratiolen=r, history = False)
dyn = compart_dyn.evolv(fraction = True, lamcond=True, xcond=True, carryingcaplim=False)
hist_evol = np.array(dyn[0])
x = dyn[1]
lam = dyn[2]
time = [i for i in range(len(hist_evol))]

compart_dyn_theo = theory_compart(T, K, d, 9.E-1, 0., 1.E-1, 8., repet, alpha, gamma, mu, nu, ratiolen=r, ncomp=int(cells))
hist_evol_theo = compart_dyn_theo.evol()[0]
time_theo = [i for i in range(len(hist_evol_theo))]

cd = os.getcwd()
np.save(cd + '/data/hist_evol_slowmut_hosttopara'+str(r)+'.npy', hist_evol)
np.save(cd + '/data/lamda_slowmut_hosttopara'+str(r)+'.npy', lam)
np.save(cd + '/data/x_slowmut_hosttopara'+str(r)+'.npy', x)
np.save(cd + '/data/time_slowmut_hosttopara'+str(r)+'.npy', time)
np.save(cd + '/data/hist_evol_theo_slowmut_hosttopara'+str(r)+'.npy', hist_evol_theo)
np.save(cd + '/data/time_theo_slowmut_hosttopara'+str(r)+'.npy', time_theo)