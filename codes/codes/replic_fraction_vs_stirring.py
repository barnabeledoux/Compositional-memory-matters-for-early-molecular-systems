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

repet = int(1.2E2)
cells = 8.E6

if len(sys.argv) > 1:
    K = float(sys.argv[1])  # The first argument after the script name
    d = float(sys.argv[2])
    r = float(sys.argv[3])
    alphaT = float(sys.argv[4])

    # Example: Handling the input as a numeric value
    try:
        print(f"Interpreted as a number: K = {K}, d = {d}, r = {r} and alpha T = {alphaT}")
    except ValueError:
        print("The input is not a numeric value.")
else:
    print("Please provide a value as a command-line argument.")
    print("Usage: python user_variable.py <value>")

slist = np.linspace(0., 1., 40)
xlist = []
alpha = np.array([[5.E-1,0.],[0.,2.E0]])
gamma = np.array([[7.E0,0.],[0.,2.E1]])
mu = np.array([1.E-2,0.]) #np.array([1.E-2,0.]) or np.array([0.,0.])
nu = np.array([1.E-2, 1.E-2]) #np.array([1.E-2,1.E-2]) or np.array([0.,0.])

T=alphaT*K/(alpha[0][0]*(K+1))

start = int(9.E1)

for s in tqdm(slist):
    compart_dyn = theory_compart_stir(T, K, d, 9.E-1, 0., 1.E-1, 8, repet, alpha, gamma, mu, nu, ratiolen = r, s=s, ncomp = int(cells))
    dyn = compart_dyn.evol(fraction = True)
    hist_evol = np.array(dyn)
    x = sum([sum(dyn[n][0]) for n in range(start,repet)])/(repet-start) if sum(dyn[-1][0]) > 0. else 0.
    xlist.append(x)

xlist = np.array(xlist)
np.save(cd + '/data/xlist_vs_s'+str(K)+str(d)+str(r)+str(alphaT)+'.npy', xlist)