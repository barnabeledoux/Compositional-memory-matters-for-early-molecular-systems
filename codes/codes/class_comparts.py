import numpy as np
import matplotlib.pyplot as plt
import random as rand
import scipy
from scipy.optimize import fsolve
import os

## Model A ##

class GillespieA:
    def __init__(self, T, K, repet, N, xini, lamdaini, alpha, gamma, mu, nu, mmutmax, ymutmax, dilu, ratiolen = 4, kappa = [], tau = [], mutantfirst=False, history=False, begin=[]):
        """
        Class for the transient compartmentalization dynamics with mutations and selection, without stirring (complete pooling) (Model A)
        T : maturation time
        K : carrying capacity
        repet : number of repetitions of the whole process
        N : number of compartments
        xini : initial fraction of WT replicases in the pool
        lamdaini : initial parameter of the poisson distribution for the number of individuals in each compartment
        alpha : matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        gamma : matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        mu : vector of the mutation rates for replicases
        nu : vector of the mutation rates for parasites
        mmutmax : number of mutant types for replicases
        ymutmax : number of mutant types for parasites
        dilu : dilution factor after pooling
        ratiolen : ratio of lengths between parasites and replicases (default 4)
        kappa : matrix of the reproduction rates for replicases (template based uncatalyzed reactions) first order (default 0)
        tau : matrix of the reproduction rates for parasites (template based uncatalyzed reactions) first order (default 0)
        mutantfirst : if True, start with only mutants (default False)
        history : if True, keep the history of each compartment (default False)
        begin : if not empty, start with the given repartition of mutants for replicases and parasites (default [])
        """
        self.T = T #Maturation time
        self.K = K #Carrying capacity
        self.repet = repet #Number of repetitions of the whole process
        self.ratiolen = ratiolen
        self.N = N #Number of compartments
        self.x = xini #initial repartition of replicases and mutants
        self.lamda = lamdaini #Parameter of the poisson distribution
        self.alpha = alpha #Matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        if len(kappa)==0:
            self.kappa = [0. for i in range(len(mu))]
        else:
            self.kappa = kappa #Matrix of the reproduction rates for replicases (template based uncatalyzed reactions) first order
        self.gamma = gamma #Matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        if len(tau)==0:
            self.tau = [0. for i in range(len(mu))]
        else:
            self.tau = tau #Matrix of the reproduction rates for parasites (template based uncatalyzed reactions) first order
        self.mu=mu #Vector of the mutation rates for replicases
        self.nu=nu #Vector of the mutation rates for parasites
        if (self.mu[-1]!=0. or self.nu[-1]!=0):
            raise Exception("The final mutation rate should be 0.")
        self.mmutantmax=mmutmax #Number of mutant types for replicases
        self.ymutantmax=ymutmax #Number of mutant types for parasites
        self.mmutant=np.zeros(mmutmax) #create the array of the frequencies of each mutant for replicases
        self.ymutant=np.zeros(ymutmax) #create the array of the frequencies of each mutant for parasites
        self.mmutant[0]=xini #Intitially we only have non-muted replicases
        self.ymutant[0]=1-xini #Intitially we only have non-muted parasites
        if len(begin)>=1:
            for i in range(mmutmax):
                self.mmutant[i] = begin[i]
            for i in range(ymutmax):
                self.ymutant[i] = begin[i+mmutmax]
        if mutantfirst:
            self.mmutant[0]=0. #Intitially we only have non-muted replicases
            self.ymutant[0]=0. #Intitially we only have non-muted parasites
            self.mmutant[1]=xini #Intitially we only have muted replicases
            self.ymutant[1]=1-xini #Intitially we only have muted parasites
        self.xini=xini
        self.dilu = dilu
        self.history = history
        self.begin = begin
        
    def reinit(self):
        self.mmutant=np.zeros(self.mmutantmax) 
        self.ymutant=np.zeros(self.ymutantmax) 
        self.mmutant[0]=self.xini #Reinitialize the fractions of each mutant replicase
        self.ymutant[0]=1-self.xini #Reinitialize the fractions of each mutant parasite
        if len(self.begin)>=1:
            for i in range(self.mmutantmax):
                self.mmutant[i] = self.begin[i]
            for i in range(self.ymutantmax):
                self.ymutant[i] = self.begin[i+self.mmutantmax]
        
    def compart(self): #Create the compartments from the pool (step 1 of the transient process), following the repartition in the pool
        k=0
        self.comp=[]
        while k<self.N: #k is the updated number of compartments, N the total number of compartments
            n=np.random.poisson(self.lamda) #Pick the total number in the compartment according to a Poisson distribution
            marr=np.zeros(len(self.mmutant)) #Array of the numbers of different mutants of replicases
            yarr=np.zeros(len(self.ymutant)) #Array of the numbers of different mutants of parasites
            repart=np.concatenate((np.array(self.mmutant), np.array(self.ymutant))) #Fractions of each mutants (first the replicases then the parasites)
            for j in range(n): #Attribute the n available slots according to the fraction of each species in the pool
                p = np.random.rand() #Pick a random number between 0 and 1
                q=0
                l=0
                while q<p:
                    q+=repart[l] #Add the fraction of each species
                    l+=1
                l-=1
                if l<len(self.mmutant):
                    marr[l]+=1
                else:
                    yarr[l-len(self.mmutant)]+=1
            self.comp.append([marr,yarr,[-1.,0.],sum(marr)+sum(yarr),marr.copy(),yarr.copy(),0]) #self.comp contains a list with 2 arrays for each of the N compartments, 1 for the mutants of replicases, 1 for the mutants of parasites. [-1,0] represents the internal clock (previous and current times) of the compartment (initially), we store the initial distribution and the number of steps
            k+=1

    def step(self, comp, carryingcap = False): #Maturation of one of the different compartments (Gillespie simulation of the parallel evolutions in each compartment)
        S = 0.
        marr,yarr,tintm1,tint,nini=comp[0].copy(),comp[1].copy(),comp[2][0],comp[2][1],comp[3] #Initialize with the number of individuals for every mutation of parasites and replicases depending on the initial repartition in the compartment
        if not(carryingcap): #If the compartment has not reached the carrying capacity yet
            for i in range(self.mu.size):
                for j in range(self.mu.size):
                    S += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
            for i in range(self.mu.size):
                for j in range(self.nu.size):
                    S += ( self.tau[j] + marr[i] * self.gamma[i,j] ) * yarr[j] #replication of parasite using replicase
        for i in range(self.mu.size):
            S+= marr[i] * self.mu[i] #mutation of a replicase
        for j in range(self.nu.size):
            S+= yarr[j] * self.nu[j] #mutation of a parasite
        x1 = np.random.random()
        x2 = np.random.random()
        tau = np.log(1 / x1) / S #Random time of the step
        R = 0.
        i = 0
        j = 0
        imut = 0
        jmut = 0
        mut=False
        iplus, jplus = False, False
        while R < x2 * S: #Determines which reaction occurs
            if not(carryingcap):
                if ((i < self.mu.size - 1) and (j <= self.mu.size - 1)):
                    R += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
                    i += 1
                    iplus, jplus = True, False
                elif ((i == self.mu.size - 1) and (j <= self.mu.size - 1)):
                    R += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
                    i = 0
                    j += 1
                    iplus, jplus = False, True
                elif ((i < self.mu.size - 1) and (j > self.mu.size - 1) and (j < self.mu.size + self.nu.size)):
                    jbis = j - self.mu.size
                    R += ( self.tau[jbis] + marr[i] * self.gamma[i,jbis] ) * yarr[jbis] #replication of parasite using replicase
                    i += 1
                    iplus, jplus = True, False
                elif ((i == self.mu.size - 1) and (j > self.mu.size - 1) and (j < self.mu.size + self.nu.size)):
                    jbis = j - self.mu.size
                    R += ( self.tau[jbis] + marr[i] * self.gamma[i,jbis] ) * yarr[jbis] #replication of parasite using replicase
                    i = 0
                    j += 1
                    iplus, jplus = False, True
                else: #A mutation occurs
                    mut=True
                    if imut <= self.mu.size -1:
                        R += marr[imut] * self.mu[imut] #mutation of a replicase
                        imut += 1
                        imutplus, jmutplus = True, False
                    else:
                        R += yarr[jmut] * self.nu[jmut] #mutation of a parasite
                        jmut += 1
                        imutplus, jmutplus = False, True
            else : # If the compartment has reached the carrying capacity, only mutations can occur
                mut=True
                if imut <= self.mu.size -1:
                    R += marr[imut] * self.mu[imut] #mutation of a replicase
                    imut += 1
                    imutplus, jmutplus = True, False
                else:
                    R += yarr[jmut] * self.nu[jmut] #mutation of a parasite
                    jmut += 1
                    imutplus, jmutplus = False, True
        comp[2][0] = tint
        tint += tau
        comp[2][1] = tint
        if mut:
            return [i-iplus,j-jplus,imut-imutplus,jmut-jmutplus,mut] #if a replicase replicates, j<self.mu.size and mut=False, if a parasite replicates, j>=self.mu.size and mut=False, if a replicase mutates, jmut=0 and mut=True, if a parasite mutates imut=self.mu.size - 1
        else:
            return [i-iplus,j-jplus,imut,jmut,mut]
        
    def allstep(self): #can be improved with Michele's method probably, perform the Gillespie step in each compartment
        for p, comp in enumerate(self.comp):
            marr,yarr,tintm1,tint,nini=comp[0].copy(),comp[1].copy(),comp[2][0],comp[2][1],comp[3]
            if sum(comp[0])>0: #If the compartment is not empty
                carryingcap = ((sum(comp[0])+sum(comp[1])/self.ratiolen)>=self.K+nini)
                if (((sum(comp[0])+sum(comp[1]/self.ratiolen))<self.K+nini) and (tint<self.T)): #If the compartment has not reached its carrying capacity yet and the maturation time has not been exceeded
                    comp[-1] += 1
                    res=self.step(comp, carryingcap) #res=[i,j,imut,jmut,mut]
                    i,j,imut,jmut,mut = res[0], res[1], res[2], res[3], res[4]
                    ##Update the populations##
                    if comp[2][1]<self.T: #If the process has time to occur
                        if mut: #If a mutation has been picked
                            if (imut >= self.mu.size): #If a parasite has muted
                                if ((comp[4][1]==0 or comp[5][1]==0) and comp[4][0]>0 and comp[5][0]>0):
                                    self.ymutcount[0] += 1
                                else:
                                    self.ymutcount[1] += 1
                                yarr[jmut+1]+=1 #The number of next type of mutant is increased by 1
                                yarr[jmut]-=1 #The number of next type of mutant is decreased by 1
                            else : #If a replicase has muted
                                if (comp[4][1]==0 and comp[4][0]>0 and comp[5][0]==0):
                                    self.mmutcount[0] += 1
                                else:
                                    self.mmutcount[1] += 1
                                marr[imut+1]+=1 #The number of next type of mutant is increased by 1
                                marr[imut]-=1 #The number of next type of mutant is decreased by 1
                        else: #If a replication has been picked
                            jbis = j - self.mu.size
                            ##print(j,jbis)
                            if  (j > self.mu.size - 1):
                                yarr[jbis] += 1 #Increase the population of parasites by 1 (doesn't depend on the replicase i used)
                            else:
                                marr[j] += 1 #Increase the population of replicases by 1 (doesn't depend on the replicase i used)
                    else:
                        comp[2][1]=self.T
                elif (tint<self.T): #If the carrying capacity has been reached but not the maturation time, mutations can still occur 
                    comp[-1] += 1
                    res=self.step(comp, carryingcap) #res=[i,j,imut,jmut,mut]
                    i,j,imut,jmut,mut = res[0], res[1], res[2], res[3], res[4]
                    ##Update the populations##
                    #We know that a mutation has been picked
                    if comp[2][1]<self.T:
                        if (imut >= self.mu.size): #If a parasite has muted
                            if ((comp[4][1]==0 or comp[5][1]==0) and comp[4][0]>0 and comp[5][0]>0):
                                self.ymutcount[0] += 1
                            else:
                                self.ymutcount[1] += 1
                            yarr[jmut+1]+=1 #The number of next type of mutant is increased by 1
                            yarr[jmut]-=1 #The number of next type of mutant is decreased by 1
                        else : #If a replicase has muted
                            if (comp[4][1]==0 and comp[4][0]>0 and comp[5][0]==0):
                                self.mmutcount[0] += 1
                            else:
                                self.mmutcount[1] += 1
                            marr[imut+1]+=1 #The number of next type of mutant is increased by 1
                            marr[imut]-=1 #The number of next type of mutant is decreased by 1
                    else:
                        comp[2][1]=self.T
            comp[0]=marr
            comp[1]=yarr
            self.comp[p]=comp
        
    def pooling(self): #After the maturation step, perform the pooling again by emptying the compartments in a common pool
        self.mmutant=np.zeros(self.mu.size) #initiate the new fractions of replicases
        self.ymutant=np.zeros(self.nu.size) #initiate the new fractions of parasites
        ntot=0 #Total number of individuals in the pool 
        for comp in self.comp: #Empty each compartment one by one
            for i in range(comp[0].size):
                ntot += comp[0][i] #Increase the total number of individuals in the pool
                self.mmutant[i] += comp[0][i] #Update the number of mutants of type i (replicases) in the pool
            for j in range(comp[1].size):
                ntot += comp[1][j] #Increase the total number of individuals in the pool
                self.ymutant[j] += comp[1][j] #Update the number of mutants of type i (parasites) in the pool
        self.lamda = (sum(self.mmutant)+sum(self.ymutant))/self.N/self.dilu
        self.mmutant = [self.mmutant[i]/ntot for i in range(comp[0].size)] #Creates the fractions of each mutant of replicases
        self.ymutant = [self.ymutant[i]/ntot for i in range(comp[1].size)] #Creates the fractions of each mutant of parasites

    def evolv(self, lamcond=False, xcond=False, carryingcaplim = True):
        self.reinit()
        repart = [[self.mmutant.tolist(),self.ymutant.tolist()]]
        lam=[self.lamda]
        x=[sum(self.mmutant)]
        x0ano, x1ano, z0ano, z1ano = 0, 0, 0, 0
        ntot = 0
        for k in range(self.repet): #Repeat the process repet times
            self.mmutcount = [0,0]
            self.ymutcount = [0,0]
            comp_hist=[[] for i in range(int(self.N))]
            if (sum(self.ymutant)<1.): #If there still are replicases
                self.compart()
                t=0
                condition_end=False
                while not(condition_end): #while t<self.T:
                    ntot = 0
                    self.allstep()
                    condition_end=True
                    tlist=[]
                    for i,comp in enumerate(self.comp):
                        if self.history:
                            comp_hist[i].append([comp[0],comp[1]])
                        tintm1, tint, nini=comp[2][0], comp[2][1], comp[3]
                        tlist.append(tint)
                        ntot += 1.
                        if carryingcaplim : #If the evolution is limited by the carrying capacity
                            if ((sum(comp[0])+sum(comp[1])/self.ratiolen)<self.K+nini and sum(comp[0])>0 and tint<self.T):
                                former_cond = condition_end
                                condition_end = False
                        else : #If the evolution is a limited by a time of maturation
                            if ((tint<self.T) and sum(comp[0])>0 and (tint != tintm1)):
                                condition_end = False
            self.pooling()
            repart.append([self.mmutant,self.ymutant])
            lam.append(self.lamda)
            x.append(sum(self.mmutant))
        if lamcond:
            if xcond:
                return [repart,x,lam]
            else:
                return [repart,lam]
        else:
            if xcond:
                return [repart,x]
            else:
                return repart
            
### Theoretical description ###

class theory_compartA:
    def __init__(self, T, K, d, x0ini, x1ini, z0ini, lamdaini, repet, alpha, gamma, mu, nu, ratiolen = 4, ncomp = 1E3, th= 1E-9):
        """
        Class for the theoretical description of transient compartmentalization dynamics with mutations and selection, within a deterministic assumtion, without stirring (complete pooling) (Model A)
        T : maturation time
        K : carrying capacity
        d : dilution factor after pooling
        x0ini : initial fraction of WT replicases in the pool
        x1ini : initial fraction of mutant replicases in the pool
        z0ini : initial fraction of WT parasites in the pool
        lamdaini : initial parameter of the poisson distribution for the number of individuals in each compartment
        repet : number of repetitions of the whole process
        alpha : matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        gamma : matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        mu : vector of the mutation rates for replicases
        nu : vector of the mutation rates for parasites
        ratiolen : ratio of lengths between parasites and replicases (default 4)
        ncomp : number of compartments (default 1E3)
        th : threshold for fraction below which a species is considered extinct (default 1E-9)

        """
        self.T = T #In hours
        self.K = K
        self.d = d
        self.ratiolen = ratiolen
        self.mmutant = np.array([x0ini,x1ini])
        self.ymutant = np.array([z0ini,1-x0ini-x1ini-z0ini])
        self.x0 = x0ini
        self.x1 = x1ini
        self.z0 = z0ini
        self.z1 = 1-x0ini-x1ini-z0ini
        self.lamda = lamdaini
        self.repet = repet
        self.mu = mu #In 1/h
        self.nu = nu #In 1/h
        self.gamma = gamma #In 1/h
        self.alpha = alpha #In 1/h
        self.fastmut = False
        self.smartmut = False
        if (self.gamma[0][1] > 0.):
            ##print('smartmut = ', self.smartmut)
            self.smartmut = True
        self.ncomp = ncomp
        self.th = th
        
    def x0star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*( (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.) + np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) )#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        m0starnomut = K*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)+lam*x0
        #mutations
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        m0starmutfav = - ( ( (self.mu[0]/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + (self.mu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)  + (self.mu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*z1)*np.exp(lam*x0))
        m0starmutunfav = - self.T*self.mu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*(x1+z0))-1)
        if x0 == 0. or self.mu[0]==0.: #Case when there is no mutations of replicases
            m0starmutfav = 0.
            m0starmutunfav = 0.
        value = max((m0starnomut + m0starmutfav + m0starmutunfav)/nstar,0.)
        self.m0 = value*nstar
        if value < self.th :
            value = 0
        return value
    
    def x1star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        self.n = lam
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*( (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.) + np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) )#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        m1starnomut = lam*x1 + K*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0) + np.exp(lam*z0) - 1.)
        #mutations
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        m1starmutfav = ( ( (self.mu[0]/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + (self.mu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)  + (self.mu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*z1)*np.exp(lam*x0))
        m1starmutunfav = self.T*self.mu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*(x1+z0))-1.)
        if x0 == 0. or self.mu[0]==0.: #Case when there is no mutations of replicases
            m1starmutfav = 0.
            m1starmutunfav = 0.
        else:
            self.mmutcount[0] += m1starmutfav
            self.mmutcount[1] += m1starmutunfav
        value = max((m1starnomut + m1starmutfav + m1starmutunfav)/nstar,0.)
        self.m1 = value*nstar
        if value < self.th :
            value = 0
        return value
    
    def z0star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*( (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.) + np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) )#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        y0starnomut = lam*z0 + K*self.ratiolen*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.)
        #mutations
        T = self.T
        y0starmutfav = -self.nu[0]*lam*z0*((K*self.ratiolen+1.)**(1.-self.alpha[0][0]/self.gamma[0][0])-1.)/(self.gamma[0][0]-self.alpha[0][0])-self.nu[0]*K*self.ratiolen*(self.alpha[0][0]*T-(1./K*self.ratiolen)**(self.gamma[0][0]/self.alpha[0][0]))*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.)/self.alpha[0][0] - self.nu[0]*lam*z0*(self.alpha[0][0]*T-(1./K*self.ratiolen)**(self.gamma[0][0]/self.alpha[0][0]))*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.)/self.alpha[0][0]
        y0starmutunfav = -self.nu[0]*T*lam*z0*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*(x1+z1)) + (np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1.))
        if z0 == 0. or self.nu[0]==0.: #Case when there is no mutations of replicases
            y0starmutfav = 0.
            y0starmutunfav = 0.
            x1 = 0.
            z1 = 0.
        value = max((y0starnomut+y0starmutfav+y0starmutunfav)/nstar,0.)
        self.y0 = value*nstar
        if value < self.th :
            value = 0
        return value
    
    def z1star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*( (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.) + np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) )#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        y1starnomut = lam*z1 + K*self.ratiolen*np.exp(-lam)*np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.)
        #mutations
        T = self.T
        y1starmutfav = self.nu[0]*lam*z0*((K*self.ratiolen+1.)**(1.-self.alpha[0][0]/self.gamma[0][0])-1.)/(self.gamma[0][0]-self.alpha[0][0]) + self.nu[0]*K*self.ratiolen*(self.alpha[0][0]*T-(1./K*self.ratiolen)**(self.gamma[0][0]/self.alpha[0][0]))*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.)/self.alpha[0][0] + self.nu[0]*lam*z0*(self.alpha[0][0]*T-(1./K*self.ratiolen)**(self.gamma[0][0]/self.alpha[0][0]))*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.)/self.alpha[0][0]
        y1starmutunfav = self.nu[0]*T*lam*z0*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*(x1+z1)) + (np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1.))
        if z0 == 0. or self.nu[0]==0.: #Case when there is no mutations of replicases
            y1starmutfav = 0.
            y1starmutunfav = 0.
        else:
            self.ymutcount[0] += y1starmutfav
            self.ymutcount[1] += y1starmutunfav        
        value = max((y1starnomut + y1starmutfav + y1starmutunfav)/nstar,0.)
        self.y1 = value*nstar
        if value < self.th :
            value = 0
        return value
    
    def lamdastar(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*( (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.) + np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) )#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        self.n = nstar
        return nstar/d
    
    def x0noise(self):
        x0, x1, z0, z1, K, d, lam, ncomp = self.x0, self.x1, self.z0, self.z1, self.K, self.d, self.lamda, self.ncomp
        emlam = np.exp(-lam)
        meanm0 = K*emlam*np.exp(lam*z1)*(np.exp(lam*x0) - 1.) + lam*x0
        meann = K*emlam*np.exp(lam*(z0 + z1))*(np.exp(lam*(x0+x1)) - 1.) + lam
        meanm02 = (K**2)*emlam*np.exp(lam*z1)*(np.exp(lam*x0) - 1.) + 2*lam*x0*K*emlam*np.exp(lam*(x0+z1))+lam*x0*(1+lam*x0)
        meann2 = (K**2)*emlam*np.exp(lam*(z0 + z1))*(np.exp(lam*(x0+x1)) - 1.) + 2*K*lam*emlam*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1+x0+x1)+lam*(1. + lam)
        covarm0n = (K**2)*emlam*np.exp(lam*z1)*(np.exp(lam*x0) - 1.) + lam*K*emlam*(x0*(1+np.exp(lam*(x0+z1)))+z1*np.exp(lam*z1)*(np.exp(lam*x0) - 1.)) + lam*x0*(1. + lam)
        return np.sqrt((1/ncomp)*((meanm0/meann)**2)*(meanm02/(meanm0)**2+meann2/(meann)**2-2*covarm0n/(meanm0*meann)))
    
    def x1noise(self):
        x0, x1, z0, z1, K, d, lam, ncomp = self.x0, self.x1, self.z0, self.z1, self.K, self.d, self.lamda, self.ncomp
        emlam = np.exp(-lam)
        meanm1 = K*emlam*(np.exp(lam*x1) - 1.)*(np.exp(lam*x0) + np.exp(lam*z0) - 1.) + lam*x1
        meann = K*emlam*np.exp(lam*(z0 + z1))*(np.exp(lam*(x0+x1)) - 1.) + lam
        meanm12 = (K**2)*emlam*(np.exp(lam*x1) - 1.)*(np.exp(lam*x0) + np.exp(lam*z0) - 1.) + 2*lam*x1*K*emlam*np.exp(lam*x1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1.) + lam*x1*(1+lam*x1)
        meann2 = (K**2)*emlam*np.exp(lam*(z0 + z1))*(np.exp(lam*(x0+x1)) - 1.) + 2*K*lam*emlam*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1+x0+x1)+lam*(1+lam)
        covarm1n = (K**2)*emlam*(np.exp(lam*x1) - 1.)*(np.exp(lam*x0) + np.exp(lam*z0) - 1.) + lam*K*emlam*(x0*np.exp(lam*x0)*(np.exp(lam*x1) - 1.) + x1*(1 + np.exp(lam*z1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1.)) + z0*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)) + lam*x1*(1. + lam)
        return np.sqrt((1/ncomp)*((meanm1/meann)**2)*(meanm12/(meanm1)**2+meann2/(meann)**2-2*covarm1n/(meanm1*meann)))
    
    def z0noise(self):
        x0, x1, z0, z1, K, d, lam, ncomp = self.x0, self.x1, self.z0, self.z1, self.K, self.d, self.lamda, self.ncomp
        emlam = np.exp(-lam)
        meany0 = K*emlam*(np.exp(lam*x0) - 1.)*(np.exp(lam*z0) - 1.)*(np.exp(lam*x1) + np.exp(lam*z1) - 1.) + lam*z0
        meann = K*emlam*np.exp(lam*(z0 + z1))*(np.exp(lam*(x0+x1)) - 1.) + lam
        meany02 = (K**2)*emlam*(np.exp(lam*x0) - 1.)*(np.exp(lam*z0) - 1.)*(np.exp(lam*x1) + np.exp(lam*z1) - 1.) + 2*lam*z0*K*emlam*np.exp(lam*z0)*(np.exp(lam*x0) - 1)*(np.exp(lam*x1) + np.exp(lam*z1) - 1) + lam*z0*(1+lam*z0)
        meann2 = (K**2)*emlam*np.exp(lam*(z0 + z1))*(np.exp(lam*(x0+x1)) - 1.) + 2*K*lam*emlam*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1+x0+x1)+lam*(1+lam)
        covary0n = (K**2)*emlam*(np.exp(lam*x0) - 1.)*(np.exp(lam*z0) - 1.)*(np.exp(lam*x1) + np.exp(lam*z1) - 1.) + lam*K*emlam*(x0*np.exp(lam*x0)*(np.exp(lam*z0) - 1.)*(np.exp(lam*x1) + np.exp(lam*z1) - 1.) + x1*np.exp(lam*x1)*(np.exp(lam*x0) - 1.)*(np.exp(lam*z0) - 1.) + z0*((np.exp(lam*x0) - 1)*(np.exp(lam*x1)+2*np.exp(lam*z1)-1) + np.exp(lam*(x0+z1))*(np.exp(lam*x1) - 1) + (np.exp(lam*x0) - 1)*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1)) + z1*np.exp(lam*z1)*(np.exp(lam*x0) - 1.)*(np.exp(lam*z0) - 1.)) + lam*z0*(1. + lam)
        return np.sqrt((1/ncomp)*((meany0/meann)**2)*(meany02/(meany0)**2+meann2/(meann)**2-2*covary0n/(meany0*meann)))
    
    def z1noise(self):
        x0, x1, z0, z1, K, d, lam, ncomp = self.x0, self.x1, self.z0, self.z1, self.K, self.d, self.lamda, self.ncomp
        emlam = np.exp(-lam)
        meany1 = K*emlam*np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1) + lam*z1
        meann = K*emlam*np.exp(lam*(z0 + z1))*(np.exp(lam*(x0+x1)) - 1.) + lam
        meany12 = (K**2)*emlam*np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1) + 2*lam*z1*K*emlam*np.exp(lam*(x0+z0+z1))*(np.exp(lam*x1) - 1.) + lam*z1*(1+lam*z1)
        meann2 = (K**2)*emlam*np.exp(lam*(z0 + z1))*(np.exp(lam*(x0+x1)) - 1.) + 2*K*lam*emlam*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1+x0+x1)+lam*(1+lam)
        covary1n = (K**2)*emlam*np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1) + lam*K*emlam*(x0*np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1) + x1*np.exp(lam*(x0+x1+z0))*(np.exp(lam*z1) - 1) + z0*np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1) + z1*np.exp(lam*(z0+z1))*(np.exp(lam*x0)*(2*np.exp(lam*x1) - 1) - 1)) + lam*z1*(1. + lam)
        return np.sqrt((1/ncomp)*((meany1/meann)**2)*(meany12/(meany1)**2+meann2/(meann)**2-2*covary1n/(meany1*meann)))
    
    def update(self):
        x0, x1, z0, z1, lamda = self.x0star(), self.x1star(), self.z0star(), self.z1star(), self.lamdastar()
        self.x0 = x0
        if self.mu[0]>0.:
            self.x1 = x1
        else:
            self.x1 = 0.
        self.z0 = z0
        if self.nu[0]>0.:
            self.z1 = z1
        else:
            self.z1 = 0.
        self.lamda = lamda
        self.mmutant = np.array([x0, x1])
        self.ymutant = np.array([z0, z1])
    
    def evol(self, noise = False, lambdaarr=False):
        hist_evol = [[self.mmutant, self.ymutant]]
        hist_evol_numb = []
        lambda_arr = []
        if noise:
            noise_evol = [[[0.,0.], [0.,0.]]]
        for k in range(self.repet):
            self.mmutcount = [0,0]
            self.ymutcount = [0,0]
            mcop, ycop = self.mmutant.copy(), self.ymutant.copy()
            self.update()
            hist_evol_numb.append([[self.m0, self.m1],[self.y0, self.y1]])
            hist_evol.append([self.mmutant,self.ymutant])
            if noise:
                noise_evol.append(np.array([[self.x0noise(), self.x1noise()],[self.z0noise(), self.z1noise()]]))
            elif lambdaarr:
                lambda_arr.append(self.lamda)
            if self.x0+self.x1+self.z0+self.z1>1.1:
                hist_evol.append([mcop,ycop])
        if noise:
            return np.array(hist_evol), np.array(hist_evol_numb), np.array(noise_evol)
        elif lambdaarr:
            return np.array(hist_evol), np.array(hist_evol_numb), np.array(lambda_arr)
        else:
            return np.array(hist_evol), np.array(hist_evol_numb)
            
### Theoretical description WITH STIRRING ###
class theory_compartA_stir:
    def __init__(self, T, K, d, x0ini, x1ini, z0ini, lamdaini, repet, alpha, gamma, mu, nu, ratiolen = 4, s=1., ncomp = int(1E3), randomstir = False):
        """
        Class for the theoretical description of transient compartmentalization dynamics with mutations and selection, within a deterministic assumtion, with stirring (Model A)
        T : maturation time
        K : carrying capacity
        d : dilution factor after pooling
        x0ini : initial fraction of WT replicases in the pool
        x1ini : initial fraction of mutant replicases in the pool
        z0ini : initial fraction of WT parasites in the pool
        lamdaini : initial parameter of the poisson distribution for the number of individuals in each compartment
        repet : number of repetitions of the whole process
        alpha : matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        gamma : matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        mu : vector of the mutation rates for replicases
        nu : vector of the mutation rates for parasites
        ratiolen : ratio of lengths between parasites and replicases (default 4)
        s : stirring parameter (1. means no stirring, <1 means mixing the compartments with the rest of the system)
        ncomp : number of compartments (default 1E3)
        randomstir : if True, the stirring parameter s is randomly chosen at each round between 0.98 and 1 (default False)
        """
        self.T = T #In hours
        self.K = K
        self.d = d
        self.ratiolen = ratiolen
        self.mmutant = np.array([x0ini,x1ini])
        self.ymutant = np.array([z0ini,1-x0ini-x1ini-z0ini])
        self.x0 = x0ini
        self.x1 = x1ini
        self.z0 = z0ini
        self.z1 = 1-x0ini-x1ini-z0ini
        self.lamda = lamdaini
        self.repet = repet
        self.mu = mu #In 1/h
        self.nu = nu #In 1/h
        self.gamma = gamma #In 1/h
        self.alpha = alpha #In 1/h
        self.fastmut = False
        self.smartmut = False
        if (self.gamma[0][1] > 0.):
            self.smartmut = True
        self.s = s
        self.ncomp = ncomp
        self.randomstir = randomstir

    def fact(self, n):
        pi, k = 1, 0
        while k < n:
            k+=1
            pi*=k
        return pi
        
    def initialization(self):
        self.comp, lam = [], self.lamda
        x0, x1, z0, z1 = self.x0, self.x1, self.z0, self.z1
        fracs = [x0, x1, z0, z1]
        for i in range(self.ncomp):
            r = np.random.rand()
            poisson, n = 0., 0
            while poisson < r:
                poisson += np.exp(-lam)*(lam**n)/(self.fact(n))
                n+=1
            n-=1
            compart = [0., 0., 0., 0.]
            for j in range(n):
                r, p = np.random.rand(), 0.
                for k, frac in enumerate(fracs):
                    p += frac
                    if r<=p:
                        break
                compart[k] += 1
            self.comp.append(compart)
            
    def update(self, monit_Xbef = False):
        x0, x1, z0, z1, K, d, lam, s, ncomp = self.x0, self.x1, self.z0, self.z1, self.K, self.d, self.lamda, self.s, self.ncomp
        if monit_Xbef: #To obtain the conditional probability distribution of xtot after maturation, dilution and stirring, conditional on xtot before maturation or before stirring
            siz=80
            xini = np.zeros(ncomp)
            Prob = np.zeros((siz, siz)) #conditional probability array of xtot after maturation, dilution and stirring, conditional on xtot before stirring
            Probam = np.zeros((siz, siz)) #conditional probability array of xtot after maturation, dilution and stirring, conditional on xtot before maturation
        if self.randomstir:
            s = np.random.rand()*(1.-0.98) + 0.98 #Random stirring between 0.98 and 1
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        Sm0, Sm1, Sy0, Sy1 = 0., 0., 0., 0.
        ## Evolve the compartments ##
        for i, compart in enumerate(self.comp):
            m0i, m1i, y0i, y1i, ni = compart[0], compart[1], compart[2], compart[3], sum(compart)
            if monit_Xbef:
                xini[i] = (m0i + m1i)/(ni + (ni==0))
            if (m1i < 1) and (y0i < 1) and (m0i > 0):
                compart[0] += K
            elif (m1i > 0) and (y1i < 1) and ((m0i < 1) or (y0i < 1)):
                compart[1] += K
            elif (m0i > 0) and (y0i > 0) and ((m1i < 1) or (y1i < 1)):
                compart[2] += K*self.ratiolen
            elif (m1i > 0) and (y1i > 0):
                compart[3] += K*self.ratiolen
            #mutations
            if (m1i < 1) and (y0i < 1) and (m0i > 0):
                mut = self.mu[0]/self.alpha[0,0] * np.log(K+1) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m0i)
                compart[0] -= mut
                compart[1] += mut
            else:
                mut = T*self.mu[0]*m0i
                compart[0] -= mut
                compart[1] += mut
            if (m0i > 0) and (y0i > 0) and ((m1i < 1) or (y1i < 1)):
                mut = (self.nu[0]*y0i/(m0i*(self.gamma[0,0]-self.alpha[0,0])))*((self.ratiolen*K+1)**(1.-self.alpha[0,0]/self.gamma[0,0]) - 1.) + self.nu[0]*(K*self.ratiolen + y0i)*(self.alpha[0,0]*T - (1/(K*self.ratiolen+1))**(self.gamma[0,0]/self.alpha[0,0]))/self.alpha[0,0]
                compart[2] -= mut
                compart[3] += mut
            else:
                mut = T*self.nu[0]*y0i
                compart[2] -= mut
                compart[3] += mut
            Sm0, Sm1, Sy0, Sy1 = Sm0 + compart[0], Sm1 + compart[1], Sy0 + compart[2], Sy1 + compart[3]
            self.comp[i] = compart
        Sn = Sm0 + Sm1 + Sy0 + Sy1
        x0, x1, z0, z1 = Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0)), Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))
        self.mmutantnum, self.ymutantnum = np.array([Sm0, Sm1]), np.array([Sy0, Sy1])
        ## Dilute ##
        Ndil = ncomp//d
        remove = rand.sample(range(ncomp),int(Ndil))
        for k in remove:
            self.comp[k] = [0., 0., 0., 0.]
        self.lamda = Sn/d/ncomp
        ## Stirring ##
        Sm0, Sm1, Sy0, Sy1 = 0, 0, 0, 0
        for i, compart in enumerate(self.comp):
            m0i, m1i, y0i, y1i, ni = compart[0], compart[1], compart[2], compart[3], sum(compart) #content after maturation and before stirring
            m0i, m1i, y0i, y1i = (1-s)*m0i + s*x0*Sn/d/ncomp, (1-s)*m1i + s*x1*Sn/d/ncomp, (1-s)*y0i + s*z0*Sn/d/ncomp, (1-s)*y1i + s*z1*Sn/d/ncomp
            lami = (1-s)*ni + s*self.lamda
            lami += (lami==0.)
            x0i, x1i, z0i, z1i = m0i/lami, m1i/lami, y0i/lami, y1i/lami #fraction after maturation to draw stirring from
            if monit_Xbef:
                xbefore = x0i + x1i
            fracs = [x0i, x1i, z0i, z1i]
            r = np.random.rand()
            poisson, n = 0., 0
            while poisson < r: #draw crowding of cell i
                poisson += np.exp(-lami)*(lami**n)/(self.fact(n))
                n+=1
            n-=1
            compart = [0., 0., 0., 0.]
            for j in range(n): #draw content in cell i
                r, p = np.random.rand(), 0.
                for k, frac in enumerate(fracs):
                    p+=frac
                    if r<=p:
                        break
                compart[k] += 1
            Sm0, Sm1, Sy0, Sy1 = Sm0 + compart[0], Sm1 + compart[1], Sy0 + compart[2], Sy1 + compart[3]
            self.comp[i] = compart
            if monit_Xbef:
                xafter = compart[0]/(n + (n==0)) + compart[1]/(n + (n==0))
                indx, indy = int(xbefore*siz), int(xafter*siz)
                Probam[min(indx,siz-1), min(indy,siz-1)] += 1
                indx, indy = int(xini[i]*siz), int(xafter*siz)
                Prob[min(indx,siz-1), min(indy,siz-1)] += 1
        Sn = Sm0 + Sm1 + Sy0 + Sy1
        self.x0, self.x1, self.z0, self.z1 = Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0)), Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))
        self.mmutant, self.ymutant = np.array([Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0))]), np.array([Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))])
        if monit_Xbef:
            return np.array(Prob)/ncomp, np.array(Probam)/ncomp
    
    def evol(self, fraction = True):
        self.initialization()
        if fraction : 
            hist_evol = [[self.mmutant, self.ymutant]]
        else :
            hist_evol = []    
        for k in range(self.repet):
            self.update()
            if fraction :
                hist_evol.append([self.mmutant,self.ymutant])
            else : 
                hist_evol.append([self.mmutantnum,self.ymutantnum])
        return np.array(hist_evol)
    

 ## Model B ##

class GillespieB:
    def __init__(self, T, K, repet, N, xini, lamdaini, alpha, gamma, mu, nu, mmutmax, ymutmax, dilu, ratiolen = 4., kappa = [], tau = [], mutantfirst=False, history=False, begin=[]):
        """
        Class for the stochastic description of transient compartmentalization dynamics with mutations and selection, within a Gillespie framework (Model B)
        T : maturation time
        K : carrying capacity
        repet : number of repetitions of the whole process
        N : number of compartments
        xini : initial fraction of WT replicases in the pool
        lamdaini : initial parameter of the poisson distribution for the number of individuals in each compartment
        alpha : matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        kappa : matrix of the reproduction rates for replicases (template based uncatalyzed reactions) first order
        gamma : matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        tau : matrix of the reproduction rates for parasites (template based uncatalyzed reactions) first order
        mu : vector of the mutation rates for replicases
        nu : vector of the mutation rates for parasites
        mmutmax : number of mutant types for replicases
        ymutmax : number of mutant types for parasites
        dilu : dilution factor after pooling
        ratiolen : ratio of lengths between parasites and replicases (default 4)
        mutantfirst : if True, the initial condition is xini of the first mutant type and 1-xini of the first mutant type for parasites (default False)
        history : if True, store the history of all compartments at each step (default False)
        begin : initial condition for all the mutants, first the replicases then the parasites (default empty)
        """
        self.T = T #Maturation time
        self.K = K #Carrying capacity
        self.repet = repet #Number of repetitions of the whole process
        self.ratiolen = ratiolen
        self.N = N #Number of compartments
        self.x = xini #initial repartition of replicases and mutants
        self.lamda = lamdaini #Parameter of the poisson distribution
        self.alpha = alpha #Matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        if len(kappa)==0:
            self.kappa = [0. for i in range(len(mu))]
        else:
            self.kappa = kappa #Matrix of the reproduction rates for replicases (template based uncatalyzed reactions) first order
        self.gamma = gamma #Matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        if len(tau)==0:
            self.tau = [0. for i in range(len(mu))]
        else:
            self.tau = tau #Matrix of the reproduction rates for parasites (template based uncatalyzed reactions) first order
        self.mu=mu #Vector of the mutation rates for replicases
        self.nu=nu #Vector of the mutation rates for parasites
        self.ratiolen = ratiolen
        #if (self.mu[-1]!=0. or self.nu[-1]!=0):
        #    raise Exception("The final mutation rate should be 0.")
        self.mmutantmax=mmutmax #Number of mutant types for replicases
        self.ymutantmax=ymutmax #Number of mutant types for parasites
        self.mmutant=np.zeros(mmutmax) #create the array of the frequencies of each mutant for replicases
        self.ymutant=np.zeros(ymutmax) #create the array of the frequencies of each mutant for parasites
        self.mmutant[0]=xini #Intitially we only have non-muted replicases
        self.ymutant[0]=1-xini #Intitially we only have non-muted parasites
        if len(begin)>=1:
            for i in range(mmutmax):
                self.mmutant[i] = begin[i]
            for i in range(ymutmax):
                self.ymutant[i] = begin[i+mmutmax]
        if mutantfirst:
            self.mmutant[0]=0. #Intitially we only have non-muted replicases
            self.ymutant[0]=0. #Intitially we only have non-muted parasites
            self.mmutant[1]=xini #Intitially we only have muted replicases
            self.ymutant[1]=1-xini #Intitially we only have muted parasites
        self.xini=xini
        self.dilu = dilu
        self.history = history
        self.begin = begin
        
    def reinit(self):
        self.mmutant=np.zeros(self.mmutantmax)
        self.ymutant=np.zeros(self.ymutantmax)
        self.mmutant[0]=self.xini #Intitially we only have non-muted replicases with fraction xini
        self.ymutant[0]=1-self.xini #Intitially we only have non-muted parasites with fraction 1 - xini
        if len(self.begin)>=1:
            for i in range(self.mmutantmax):
                self.mmutant[i] = self.begin[i]
            for i in range(self.ymutantmax):
                self.ymutant[i] = self.begin[i+self.mmutantmax]
        
    def compart(self): #Create the compartments from the pool (step 1 of the transient process), following the repartition in the pool
        k=0
        self.comp=[]
        while k<self.N: #k is the updated number of compartments, N the total number of compartments
            n=np.random.poisson(self.lamda) #Pick the total number in the compartment according to a Poisson distribution
            marr=np.zeros(len(self.mmutant)) #Array of the numbers of different mutants of replicases
            yarr=np.zeros(len(self.ymutant)) #Array of the numbers of different mutants of parasites
            repart=np.concatenate((np.array(self.mmutant), np.array(self.ymutant))) #Fractions of each mutants (first the replicases then the parasites)
            for j in range(n): #Attribute the n available slots according to the fraction of each species in the pool
                p = np.random.rand() #Pick a random number between 0 and 1
                q=0
                l=0
                while q<p:
                    q+=repart[l] #Add the fraction of each species
                    l+=1
                l-=1
                if l<len(self.mmutant):
                    marr[l]+=1
                else:
                    yarr[l-len(self.mmutant)]+=1
            self.comp.append([marr,yarr,[-1.,0.],sum(marr)+sum(yarr),marr.copy(),yarr.copy(),0]) #self.comp contains a list with 2 arrays for each of the N compartments, 1 for the mutants of replicases, 1 for the mutants of parasites. [-1,0] represents the internal clock (previous and current times) of the compartment (initially), we store the initial distribution and the number of steps
            k+=1
        

    def step(self, comp, carryingcap = False): #Maturation of one of the different compartments (Gillespie simulation of the parallel evolutions in each compartment)
        S = 0.
        marr,yarr,tintm1,tint,nini=comp[0].copy(),comp[1].copy(),comp[2][0],comp[2][1],comp[3] #Initialize with the number of individuals for every mutation of parasites and replicases depending on the initial repartition in the compartment
        if not(carryingcap): #If the compartment has not reached the carrying capacity yet
            for i in range(self.mu.size):
                for j in range(self.mu.size):
                    S += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
            for i in range(self.mu.size):
                for j in range(self.nu.size):
                    S += ( self.tau[j] + marr[i] * self.gamma[i,j] ) * yarr[j] #replication of parasite using replicase
        for i in range(self.mu.size):
            S+= marr[i] * self.mu[i] #mutation of a replicase
        for j in range(self.nu.size):
            S+= marr[j] * self.nu[j] #mutation of a parasite
        x1 = np.random.random()
        x2 = np.random.random()
        tau = np.log(1 / x1) / S #time until next reaction
        R = 0.
        i = 0
        j = 0
        imut = 0
        jmut = 0
        mut=False
        iplus, jplus = False, False
        while R < x2 * S:
            if not(carryingcap):
                if ((i < self.mu.size - 1) and (j <= self.mu.size - 1)):
                    R += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
                    i += 1
                    iplus, jplus = True, False
                elif ((i == self.mu.size - 1) and (j <= self.mu.size - 1)):
                    R += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
                    i = 0
                    j += 1
                    iplus, jplus = False, True
                elif ((i < self.mu.size - 1) and (j > self.mu.size - 1) and (j < self.mu.size + self.nu.size)):
                    jbis = j - self.mu.size
                    R += ( self.tau[jbis] + marr[i] * self.gamma[i,jbis] ) * yarr[jbis] #replication of parasite using replicase
                    i += 1
                    iplus, jplus = True, False
                elif ((i == self.mu.size - 1) and (j > self.mu.size - 1) and (j < self.mu.size + self.nu.size)):
                    jbis = j - self.mu.size
                    R += ( self.tau[jbis] + marr[i] * self.gamma[i,jbis] ) * yarr[jbis] #replication of parasite using replicase
                    i = 0
                    j += 1
                    iplus, jplus = False, True
                else: #A mutation occurs
                    mut=True
                    if imut <= self.mu.size -1:
                        R += marr[imut] * self.mu[imut] #mutation of a replicase
                        imut += 1
                        imutplus, jmutplus = True, False
                    else:
                        R += marr[jmut] * self.nu[jmut] #mutation of a parasite
                        jmut += 1
                        imutplus, jmutplus = False, True
            else : # If the compartment has reached the carrying capacity, only mutations can occur
                mut=True
                if imut <= self.mu.size -1:
                    R += marr[imut] * self.mu[imut] #mutation of a replicase
                    imut += 1
                    imutplus, jmutplus = True, False
                else:
                    R += marr[jmut] * self.nu[jmut] #mutation of a parasite
                    jmut += 1
                    imutplus, jmutplus = False, True
        comp[2][0] = tint
        tint += tau
        comp[2][1] = tint
        if mut:
            return [i-iplus,j-jplus,imut-imutplus,jmut-jmutplus,mut] #if a replicase replicates, j<self.mu.size and mut=False, if a parasite replicates, j>=self.mu.size and mut=False, if a replicase mutates, jmut=0 and mut=True, if a parasite mutates imut=self.mu.size - 1
        else:
            return [i-iplus,j-jplus,imut,jmut,mut]
        
    def allstep(self): #Perform Gillespie step in each compartment
        Sm0, Sm1, Sy0, Sy1 = 0., 0., 0., 0.
        for p, comp in enumerate(self.comp):
            marr,yarr,tintm1,tint,nini=comp[0].copy(),comp[1].copy(),comp[2][0],comp[2][1],comp[3]
            if sum(comp[0])>0: #If the compartment is not empty
                carryingcap = ((sum(comp[0])+sum(comp[1])/self.ratiolen)>=self.K+nini)
                if (((sum(comp[0])+sum(comp[1])/self.ratiolen)<self.K+nini) and (tint<self.T)): #If the compartment has not reached its carrying capacity yet and the maturation time has not been exceeded
                    comp[-1] += 1
                    res=self.step(comp, carryingcap)
                    i,j,imut,jmut,mut = res[0], res[1], res[2], res[3], res[4]
                    ##Update the populations##
                    if comp[2][1]<self.T: #If the process has time to occur
                        if mut: #If a mutation has been picked
                            if (imut >= self.mu.size): #If a host has muted to parasite
                                if ((comp[4][1]==0 or comp[5][1]==0) and comp[4][0]>0 and comp[5][0]>0):
                                    self.ymutcount[0] += 1
                                else:
                                    self.ymutcount[1] += 1
                                marr[jmut]-=1 #The number of hosts is decreased by 1
                                yarr[jmut]+=1 #The number of mutant parasites is increased by 1
                            else : #If a replicase has muted
                                if (comp[4][1]==0 and comp[4][0]>0 and comp[5][0]==0):
                                    self.mmutcount[0] += 1
                                else:
                                    self.mmutcount[1] += 1
                                marr[imut+1]+=1 #The number of next type of mutant is increased by 1
                                marr[imut]-=1 #The number of next type of mutant is decreased by 1
                        else: #If a replication has been picked
                            jbis = j - self.mu.size
                            if  (j > self.mu.size - 1):
                                yarr[jbis] += 1 #Increase the population of parasites by 1 (doesn't depend on the replicase i used)
                            else:
                                marr[j] += 1 #Increase the population of replicases by 1 (doesn't depend on the replicase i used)
                    else:
                        comp[2][1]=self.T
                elif (tint<self.T): #If the carrying capacity has been reached but not the maturation time, mutations can still occur 
                    comp[-1] += 1
                    res=self.step(comp, carryingcap)
                    i,j,imut,jmut,mut = res[0], res[1], res[2], res[3], res[4]
                    ##Update the populations##
                    #We know that a mutation has been picked
                    if comp[2][1]<self.T:
                        if (imut >= self.mu.size): #If a parasite has muted
                            if ((comp[4][1]==0 or comp[5][1]==0) and comp[4][0]>0 and comp[5][0]>0):
                                self.ymutcount[0] += 1
                            else:
                                self.ymutcount[1] += 1
                            yarr[jmut]+=1 #The number of mutant parasites is increased by 1
                            marr[jmut]-=1 #The number of corresponding hosts is decreased by 1
                        else : #If a replicase has muted
                            if (comp[4][1]==0 and comp[4][0]>0 and comp[5][0]==0):
                                self.mmutcount[0] += 1
                            else:
                                self.mmutcount[1] += 1
                            marr[imut+1]+=1 #The number of next type of mutant is increased by 1
                            marr[imut]-=1 #The number of next type of mutant is decreased by 1
                    else:
                        comp[2][1]=self.T
            comp[0]=marr
            comp[1]=yarr
            Sm0 += marr[0]
            Sm1 += marr[1]
            Sy0 += yarr[0]
            Sy1 += yarr[1]
            self.comp[p]=comp
        self.mmutantnum,self.ymutantnum = [Sm0, Sm1], [Sy0, Sy1]
        
    def pooling(self): #After the maturation step, perform the pooling again by emptying the compartments in a common pool
        self.mmutant=np.zeros(self.mu.size) #initiate the new fractions of replicases
        self.ymutant=np.zeros(self.nu.size) #initiate the new fractions of parasites
        ntot=0 #Total number of individuals in the pool 
        for comp in self.comp: #Empty each compartment one by one
            for i in range(comp[0].size):
                ntot += comp[0][i] #Increase the total number of individuals in the pool
                self.mmutant[i] += comp[0][i] #Update the number of mutants of type i (replicases) in the pool
            for j in range(comp[1].size):
                ntot += comp[1][j] #Increase the total number of individuals in the pool
                self.ymutant[j] += comp[1][j] #Update the number of mutants of type i (parasites) in the pool
        self.lamda = (sum(self.mmutant)+sum(self.ymutant))/self.N/self.dilu
        self.mmutant = [self.mmutant[i]/ntot for i in range(comp[0].size)] #Creates the fractions of each mutant of replicases
        self.ymutant = [self.ymutant[i]/ntot for i in range(comp[1].size)] #Creates the fractions of each mutant of parasites

    def evolv(self, fraction = False, lamcond=False, xcond=False, carryingcaplim = True):
        self.reinit()
        repart = []
        lam=[self.lamda]
        x=[sum(self.mmutant)]
        x0ano, x1ano, z0ano, z1ano = 0, 0, 0, 0
        ntot = 0
        for k in range(self.repet): #Repeat the process repet times
            self.mmutcount = [0,0]
            self.ymutcount = [0,0]
            comp_hist=[[] for i in range(int(self.N))]
            if (sum(self.ymutant)<1.): #If there still are replicases
                self.compart()
                t=0
                condition_end=False
                while not(condition_end):
                    ntot = 0
                    self.allstep()
                    condition_end=True
                    tlist=[]
                    for i,comp in enumerate(self.comp):
                        if self.history:
                            comp_hist[i].append([comp[0],comp[1]])
                        tintm1, tint, nini=comp[2][0], comp[2][1], comp[3]
                        tlist.append(tint)
                        ntot += 1.
                        if carryingcaplim : #If the evolution is limited by the carrying capacity
                            if ((sum(comp[0])+sum(comp[1])/self.ratiolen)<self.K+nini and sum(comp[0])>0 and tint<self.T):
                                former_cond = condition_end
                                condition_end = False
                        else : #If the evolution is a limited by a time of maturation
                            if ((tint<self.T) and sum(comp[0])>0 and (tint != tintm1)):
                                condition_end = False
                for comp in self.comp: #print anomalies
                    ntot += 1
                    mini, yini = comp[4], comp[5]
                    if ((mini[0]>0) and (mini[1]==0) and (yini[0]==0)):
                        if comp[0][0]<self.K*0.8 :
                            x0ano+=1
                    elif ((mini[1]>0) and (yini[1]==0) and (mini[0]==0 or yini[0]==0)):
                        if comp[0][1]<self.K*0.8 :
                            x1ano+=1
                    elif ((mini[0]>0) and (yini[0]>0) and (mini[1]==0 or yini[1]==0)):
                        if comp[1][0]/self.ratiolen<self.K*0.8 :
                            z0ano+=1
                    elif ((mini[1]>0) and (yini[1]>0)):
                        if comp[1][1]/self.ratiolen<self.K*0.7 :
                            z1ano+=1
            self.pooling()
            if fraction:
                repart.append([self.mmutant,self.ymutant])
            else:
                repart.append([self.mmutantnum,self.ymutantnum])
            lam.append(self.lamda)
            x.append(sum(self.mmutant))
        if lamcond:
            if xcond:
                return [repart,x,lam]
            else:
                return [repart,lam]
        else:
            if xcond:
                return [repart,x]
            else:
                return repart
    
### Theoretical description WITH STIRRING ###

class theory_compart_stir:
    def __init__(self, T, K, d, x0ini, x1ini, z0ini, lamdaini, repet, alpha, gamma, mu, nu, ratiolen = 4., s=1., ncomp = int(1E3), randomstir = False, history=False):
        """
        Class for the theoretical description of transient compartmentalization dynamics with mutations and selection, within a deterministic framework with stirring (Model B)
        T : maturation time
        K : carrying capacity
        d : dilution factor after pooling
        x0ini : initial fraction of WT replicases in the pool
        x1ini : initial fraction of mutant replicases in the pool
        z0ini : initial fraction of WT parasites in the pool
        lamdaini : initial parameter of the poisson distribution for the number of individuals in each compartment
        repet : number of repetitions of the whole process
        alpha : matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        gamma : matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        mu : vector of the mutation rates for replicases
        nu : vector of the mutation rates for parasites
        ratiolen : ratio of lengths between parasites and replicases (default 4)
        s : stirring parameter
        ncomp : number of compartments
        randomstir : if True, the stirring parameter s is drawn randomly at each round (default False)
        history : if True, store the history of all compartments at each step (default False)
        """
        self.T = T #In hours
        self.K = K
        self.d = d
        self.ratiolen = ratiolen
        self.mmutant = np.array([x0ini,x1ini])
        self.ymutant = np.array([z0ini,1-x0ini-x1ini-z0ini])
        self.x0 = x0ini
        self.x1 = x1ini
        self.z0 = z0ini
        self.z1 = 1-x0ini-x1ini-z0ini
        self.lamda = lamdaini
        self.repet = repet
        self.mu = mu #In 1/h
        self.nu = nu #In 1/h
        self.gamma = gamma #In 1/h
        self.alpha = alpha #In 1/h
        self.fastmut = False
        self.smartmut = False
        if (self.gamma[0][1] > 0.):
            self.smartmut = True
        self.s = s
        self.ncomp = ncomp
        self.randomstir = randomstir
        self.history = history
    
    def fact(self, n):
        pi, k = 1, 0
        while k < n:
            k+=1
            pi*=k
        return pi
        
    def initialization(self):
        self.comp, lam = [], self.lamda
        x0, x1, z0, z1 = self.x0, self.x1, self.z0, self.z1
        fracs = [x0, x1, z0, z1]
        Sm0, Sm1, Sy0, Sy1 = 0., 0., 0., 0.
        for i in range(self.ncomp):
            r = np.random.rand()
            poisson, n = 0., 0
            while poisson < r:
                poisson += np.exp(-lam)*(lam**n)/(self.fact(n))
                n+=1
            n-=1
            compart = [0., 0., 0., 0.]
            for j in range(n):
                r, p = np.random.rand(), 0.
                for k, frac in enumerate(fracs):
                    p += frac
                    if r<=p:
                        break
                compart[k] += 1
            self.comp.append(compart)
            Sm0, Sm1, Sy0, Sy1 = Sm0 + compart[0], Sm1 + compart[1], Sy0 + compart[2], Sy1 + compart[3]
            
    def update(self, monit_Xbef=False, siz=80):
        x0, x1, z0, z1, K, d, lam, s, ncomp = self.x0, self.x1, self.z0, self.z1, self.K, self.d, self.lamda, self.s, self.ncomp
        if monit_Xbef:
            siz=siz
            xini = np.zeros(ncomp)
            Prob = np.zeros((siz, siz))
            Probam = np.zeros((siz, siz))
        if self.randomstir:
            s = np.random.rand()*(1.-0.98) + 0.98
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        Sm0, Sm1, Sy0, Sy1 = 0., 0., 0., 0.
        ## Evolve the compartments ##
        if self.history:
            hist_compbycomp_step=[]
        for i, compart in enumerate(self.comp):
            m0i, m1i, y0i, y1i, ni = compart[0], compart[1], compart[2], compart[3], sum(compart)
            if monit_Xbef:
                xini[i] = (m0i+m1i)/(ni+(ni==0))
            if (m1i < 1) and (y0i < 1) and (m0i > 0):
                compart[0] += K
            elif (m1i > 0) and (y1i < 1) and ((m0i < 1) or (y0i < 1)):
                compart[1] += K
            elif (m0i > 0) and (y0i > 0) and ((m1i < 1) or (y1i < 1)):
                compart[2] += K*self.ratiolen
            elif (m1i > 0) and (y1i > 0):
                compart[3] += K*self.ratiolen
            if self.history:
                hist_compbycomp_step.append([compart[0], compart[1], compart[2], compart[3]])
            #mutations
            if (m1i < 1) and (y0i < 1) and (m0i > 0): #if m0 is favoured
                mut = self.mu[0]/self.alpha[0,0] * np.log(K+1) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m0i)
                ##print(mut)
                compart[0] -= mut
                compart[1] += mut
                mut = self.nu[0]/self.alpha[0,0] * np.log(K+1) + self.nu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m0i)
                compart[0] -= mut
                compart[2] += mut
            else: #if m0 is not favoured
                mut = T*self.mu[0]*m0i
                ##print(mut)
                compart[0] -= mut
                compart[1] += mut
                mut = T*self.nu[0]*m0i
                compart[0] -= mut
                compart[2] += mut
            if ((m0i < 1) or (y0i < 1)) and (y1i < 1) and (m1i > 0): #if m1 is favoured
                mut = self.nu[1]/self.alpha[1,1] * np.log(K+1) + self.nu[1]*max(0,T-K/(self.alpha[1,1]*(K+1)))*(K+m1i)
                ##print(mut)
                compart[1] -= mut
                compart[3] += mut
            else: #if m1 is not favoured
                mut = T*self.nu[1]*m1i
                ##print('unfav', mut)
                compart[1] -= mut
                compart[3] += mut                
            Sm0, Sm1, Sy0, Sy1 = Sm0 + compart[0], Sm1 + compart[1], Sy0 + compart[2], Sy1 + compart[3]
            self.comp[i] = compart
        if self.history:
            self.hist_compbycomp.append(hist_compbycomp_step)
        Sn = Sm0 + Sm1 + Sy0 + Sy1
        x0, x1, z0, z1 = Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0)), Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))
        self.mmutantnum, self.ymutantnum = np.array([Sm0, Sm1]), np.array([Sy0, Sy1])
        ## Dilute ##
        Ndil = ncomp//d
        remove = rand.sample(range(ncomp),int(Ndil))
        for k in remove:
            self.comp[k] = [0., 0., 0., 0.]
        self.lamda = Sn/d/ncomp
        ##print('af dil', self.lamda, self.comp)
        ## Stirring ##
        Sm0, Sm1, Sy0, Sy1 = 0, 0, 0, 0
        for i, compart in enumerate(self.comp):
            m0i, m1i, y0i, y1i, ni = compart[0], compart[1], compart[2], compart[3], sum(compart)
            m0i, m1i, y0i, y1i = (1-s)*m0i + s*x0*Sn/d/ncomp, (1-s)*m1i + s*x1*Sn/d/ncomp, (1-s)*y0i + s*z0*Sn/d/ncomp, (1-s)*y1i + s*z1*Sn/d/ncomp
            lami = (1-s)*ni + s*self.lamda
            lami += (lami==0.)
            x0i, x1i, z0i, z1i = m0i/lami, m1i/lami, y0i/lami, y1i/lami
            if monit_Xbef:
                xbefore = x0i + x1i
            fracs = [x0i, x1i, z0i, z1i]
            r = np.random.rand()
            poisson, n = 0., 0
            while poisson < r:
                poisson += np.exp(-lami)*(lami**n)/(self.fact(n))
                n+=1
            n-=1
            compart = [0., 0., 0., 0.]
            for j in range(n):
                r, p = np.random.rand(), 0.
                for k, frac in enumerate(fracs):
                    p+=frac
                    if r<=p:
                        break
                compart[k] += 1
            Sm0, Sm1, Sy0, Sy1 = Sm0 + compart[0], Sm1 + compart[1], Sy0 + compart[2], Sy1 + compart[3]
            if monit_Xbef:
                xafter = compart[0]/(n + (n==0)) + compart[1]/(n + (n==0))
                indx, indy = int(xbefore*siz), int(xafter*siz)
                Probam[min(indx,siz-1), min(indy,siz-1)] += 1
                indx, indy = int(xini[i]*siz), int(xafter*siz)
                Prob[min(indx,siz-1), min(indy,siz-1)] += 1
            self.comp[i] = compart
        Sn = Sm0 + Sm1 + Sy0 + Sy1
        self.x0, self.x1, self.z0, self.z1 = Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0)), Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))
        self.mmutant, self.ymutant = np.array([Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0))]), np.array([Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))])
        if monit_Xbef:
            return np.array(Prob)/ncomp, np.array(Probam)/ncomp
        
    
    def evol(self, fraction = False, lambdaarr = False):
        self.initialization()
        if self.history:
            self.hist_compbycomp=[]
        hist_evol = []
        for k in range(self.repet):
            self.update()
            if fraction:
                hist_evol.append([self.mmutant,self.ymutant])
            else:
                hist_evol.append([self.mmutantnum,self.ymutantnum])   
        if self.history:
            return np.array(hist_evol), np.array(self.hist_compbycomp)
        else:
            return np.array(hist_evol)
        

### Theoretical description ###

class theory_compart:
    def __init__(self, T, K, d, x0ini, x1ini, z0ini, lamdaini, repet, alpha, gamma, mu, nu, ratiolen = 4, ncomp = 1E3):
        """
        Class for the theoretical description of transient compartmentalization dynamics with mutations and selection, within a deterministic framework without stirring (Model B)
        T : maturation time
        K : carrying capacity
        d : dilution factor after pooling
        x0ini : initial fraction of WT replicases in the pool
        x1ini : initial fraction of mutant replicases in the pool
        z0ini : initial fraction of WT parasites in the pool
        lamdaini : initial parameter of the poisson distribution for the number of individuals in each compartment
        repet : number of repetitions of the whole process
        alpha : matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        gamma : matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        mu : vector of the mutation rates for replicases
        nu : vector of the mutation rates for parasites
        ratiolen : ratio of lengths between parasites and replicases (default 4)
        ncomp : number of compartments
        """
        self.T = T #In hours
        self.K = K
        self.d = d
        self.ratiolen = ratiolen
        self.mmutant = np.array([x0ini,x1ini])
        self.ymutant = np.array([z0ini,1-x0ini-x1ini-z0ini])
        self.x0 = x0ini
        self.x1 = x1ini
        self.z0 = z0ini
        self.z1 = 1-x0ini-x1ini-z0ini
        self.lamda = lamdaini
        self.repet = repet
        self.mu = mu #In 1/h
        self.nu = nu #In 1/h
        self.gamma = gamma #In 1/h
        self.alpha = alpha #In 1/h
        self.smartmut = False
        if (self.gamma[0][1] > 0.):
            self.smartmut = True
        self.ncomp = ncomp
    
    def x0star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*(np.exp(lam*(x0+z0))*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.))#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        m0starnomut = K*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)+lam*x0
        #mutations
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        m0starmutfav = - ( ( ((self.mu[0]+self.nu[0])/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)  + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*z1)*np.exp(lam*x0))
        m0starmutunfav = - self.T*(self.mu[0]+self.nu[0])*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*(x1+z0))-1.)
        if x0 == 0. or self.mu[0]==0.: #Case when there is no mutations of replicases
            m0starmutfav = 0.
            m0starmutunfav = 0.
        if self.smartmut :
            m0star = lam * x0 + K * np.exp(-lam) * (np.exp(lam*x0) - 1.)
            m0starmutfav = - ( ( ((self.mu[0]+self.nu[0])/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)  + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0))
            m0starmutunfav = - self.T*(self.mu[0]+self.nu[0])*lam*x0*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*(x0+x1+z0))-1.)
            value = max((m0star+ m0starmutfav + m0starmutunfav)/nstar,0.)
            self.m0 = value*nstar
        else :
            value = max((m0starnomut + m0starmutfav + m0starmutunfav)/nstar,0.)
            self.m0 = m0starnomut + m0starmutfav + m0starmutunfav
        return value
    
    def x1star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        self.n = lam
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*(np.exp(lam*(x0+z0))*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.))#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        m1starnomut = lam*x1 + K*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0) + np.exp(lam*z0) - 1.)
        #mutations
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        m1starmutfav = ( ( (self.mu[0]/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + (self.mu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)  + (self.mu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*(x0+z1))) - (self.nu[1]*np.log(1/(1-self.alpha[0][0]*Tfin))/self.alpha[1,1] + self.nu[1]*K*max(0,(self.T - Tfin)))*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0)+np.exp(lam*z0)-1.) - lam*x1*self.nu[1]*max(0,(self.T - Tfin))*np.exp(-lam)*np.exp(lam*x1)*(np.exp(lam*x0)+np.exp(lam*z0)-1.)
        m1starmutunfav = self.T*self.mu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*(x1+z0))-1.) - lam*x1*self.nu[1]*self.T*np.exp(-lam)*np.exp(lam*x1)*((np.exp(lam*z1)-1.)*np.exp(lam*(x0+z0)) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.))
        if x0 == 0. or self.mu[0]==0.: #Case when there is no mutations of replicases
            m1starmutfav = 0.
            m1starmutunfav = 0.
        else:
            self.mmutcount[0] += m1starmutfav
            self.mmutcount[1] += m1starmutunfav
        if self.smartmut :
            if x1 == 0.:
                m1star = 0.
            else:
                m1star = lam * x1 + K*np.exp(-lam)*(np.exp(lam*x1) - 1.)*(np.exp(lam*x0) + np.exp(lam*z0) - 1.)
            m1starmutfav =  ( ( ((self.mu[0])/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + ((self.mu[0])*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)  + ((self.mu[0])*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0)) - (self.nu[1]*np.log(K+1)/self.alpha[1,1] + self.nu[1]*K*max(0,(self.T - Tfin)))*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0)+np.exp(lam*z0)-1.) - lam*x1*self.nu[1]*max(0,(self.T - Tfin))*np.exp(-lam)*np.exp(lam*x1)*(np.exp(lam*x0)+np.exp(lam*z0)-1.)
            m1starmutunfav = self.T*(self.mu[0])*lam*x0*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*(x0+x1+z0))-1.) - lam*x1*self.nu[1]*self.T*np.exp(-lam)*np.exp(lam*x1)*((np.exp(lam*z1)-1.)*np.exp(lam*(x0+z0)) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.))
            value = max((m1star + m1starmutfav + m1starmutunfav)/nstar,0.)
            self.m1 = value*nstar
        else :
            value = max((m1starnomut + m1starmutfav + m1starmutunfav)/nstar,0.)
            self.m1 = m1starnomut + m1starmutfav + m1starmutunfav
        return value
    
    def z0star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*(np.exp(lam*(x0+z0))*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.))#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        y0starnomut = lam*z0 + K*self.ratiolen*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.)
        #mutations
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        y0starmutfav = ( ( (self.nu[0]/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + (self.nu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)  + (self.nu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*(x0+z1)))
        y0starmutunfav = self.T*self.nu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*(x1+z0))-1)
        if z0 == 0. or self.nu[0]==0.: #Case when there is no mutations of replicases
            y0starmutfav = 0.
            y0starmutunfav = 0.
            x1 = 0.
            z1 = 0.
            value = max((y0starnomut+y0starmutfav+y0starmutunfav)/nstar,0.)
        if self.smartmut :
            y0star = lam * z0 + K*np.exp(-lam)*np.exp(lam*x1)*(np.exp(lam*x0) - 1.)*(np.exp(lam*z0) - 1.)
            y0starmutfav = ( ( ((self.nu[0])/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + ((self.nu[0])*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)  + ((self.nu[0])*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0))
            y0starmutunfav = self.T*(self.nu[0])*lam*x0*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*(x0+x1+z0))-1.)
            value = max((y0star+y0starmutfav+y0starmutunfav)/nstar,0.)
            self.y0 =  value*nstar
        else :
            self.y0 = y0starnomut+y0starmutfav+y0starmutunfav
            value = max((y0starnomut+y0starmutfav+y0starmutunfav)/nstar, 0.)
        return value
    
    def z1star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*(np.exp(lam*(x0+z0))*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.))#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        y1starnomut = lam*z1 + K*self.ratiolen*np.exp(-lam)*np.exp(lam*(x0+z0))*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.)
        #mutations
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        y1starmutfav = (self.nu[1]*np.log(K+1)/self.alpha[1,1] + self.nu[1]*K*max(0,(self.T - Tfin)))*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0)+np.exp(lam*z0)-1.) + lam*x1*self.nu[1]*max(0,(self.T - Tfin))*np.exp(-lam)*np.exp(lam*x1)*(np.exp(lam*x0)+np.exp(lam*z0)-1.)
        y1starmutunfav = lam*x1*self.nu[1]*self.T*np.exp(-lam)*np.exp(lam*x1)*((np.exp(lam*z1)-1.)*np.exp(lam*(x0+z0)) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.))
        if z0 == 0. or self.nu[0]==0.: #Case when there is no mutations of replicases
            y1starmutfav = 0.
            y1starmutunfav = 0.
        else:
            self.ymutcount[0] += y1starmutfav
            self.ymutcount[1] += y1starmutunfav
        if self.smartmut :
            if z1 == 0.:
                y1star = 0.
            else:
                y1star = lam * z1 + K*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*z1) - 1.)*(np.exp(lam*(x0+x1)) - 1.)
            value = max((y1star+y1starmutfav+y1starmutunfav)/nstar,0.)
            self.y1 = value*nstar
        else :
            self.y1 = y1starnomut + y1starmutfav + y1starmutunfav
            value = max((y1starnomut + y1starmutfav + y1starmutunfav)/nstar, 0.)
        return value
    
    def lamdastar(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        nstar = lam + K*np.exp(-lam)*(np.exp(lam*z1)*(np.exp(lam*x0)-1)+(np.exp(lam*x0)+np.exp(lam*z0)-1)*(np.exp(lam*x1)-1)) + K*self.ratiolen*np.exp(-lam)*(np.exp(lam*(x0+z0))*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)*(np.exp(lam*x1)+np.exp(lam*z1)-1.))#lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        self.n = nstar
        return nstar/d
    
    def update(self):
        x0, x1, z0, z1, lamda = self.x0star(), self.x1star(), self.z0star(), self.z1star(), self.lamdastar()
        self.x0 = x0
        if self.mu[0]>0.:
            self.x1 = x1
        else:
            self.x1 = 0.
        self.z0 = z0
        if self.nu[0]>0.:
            self.z1 = z1
        else:
            self.z1 = 0.
        self.lamda = lamda
        self.mmutant = np.array([x0, x1])
        self.ymutant = np.array([z0, z1])
    
    def evol(self, lambdaarr=False):
        hist_evol = [[self.mmutant, self.ymutant]]
        hist_evol_numb = []
        lambda_arr = []
        for k in range(self.repet):
            self.mmutcount = [0,0]
            self.ymutcount = [0,0]
            mcop, ycop = self.mmutant.copy(), self.ymutant.copy()
            self.update()
            hist_evol_numb.append([[self.m0, self.m1],[self.y0, self.y1]])
            hist_evol.append([self.mmutant,self.ymutant])
            if lambdaarr:
                lambda_arr.append(self.lamda)
            if self.x0+self.x1+self.z0+self.z1>1.1:
                hist_evol.append([mcop,ycop])
        if lambdaarr:
            return np.array(hist_evol), np.array(hist_evol_numb), np.array(lambda_arr)
        else:
            return np.array(hist_evol), np.array(hist_evol_numb)
        

## Trade offs ##

class Gillespiebet:
    def __init__(self, T, K, repet, N, xini, lamdaini, alpha, gamma, mu, nu, mmutmax, ymutmax, dilu, ratiolen=4, kappa = [], tau = [], mutantfirst=False, history=False, begin=[]):
        """
        Class for the stochastic description of transient compartmentalization dynamics with mutations and selection, within a Gillespie framework without stirring (Model B with trade offs)
        T : maturation time
        K : carrying capacity
        repet : number of repetitions of the whole process
        N : number of compartments
        xini : initial fraction of replicases in the pool
        lamdaini : initial parameter of the poisson distribution for the number of individuals in each compartment
        alpha : matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        gamma : matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        mu : vector of the mutation rates for replicases
        nu : vector of the mutation rates for parasites
        mmutmax : number of mutant types for replicases
        ymutmax : number of mutant types for parasites
        dilu : dilution factor after pooling
        ratiolen : ratio of lengths between parasites and replicases (default 4)
        kappa : matrix of the reproduction rates for replicases (template based uncatalyzed reactions) first order
        tau : matrix of the reproduction rates for parasites (template based uncatalyzed reactions) first order
        mutantfirst : if True, the initial condition is the first mutant type for both replicases and parasites (default False)
        history : if True, the history of each compartment at each step is stored (default False)
        begin : initial condition for the different types of replicases and parasites, if empty the initial condition is xini for non-mutated replicases and 1-xini for non-mutated parasites (default [])
        """
        self.T = T #Maturation time
        ##print(self.T)
        self.K = K #Carrying capacity
        self.repet = repet #Number of repetitions of the whole process
        self.ratiolen = ratiolen
        self.N = N #Number of compartments
        self.x = xini #initial repartition of replicases and mutants
        self.lamda = lamdaini #Parameter of the poisson distribution
        self.alpha = alpha #Matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        if len(kappa)==0:
            self.kappa = [0. for i in range(len(mu))]
        else:
            self.kappa = kappa #Matrix of the reproduction rates for replicases (template based uncatalyzed reactions) first order
        self.gamma = gamma #Matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        if len(tau)==0:
            self.tau = [0. for i in range(len(mu))]
        else:
            self.tau = tau #Matrix of the reproduction rates for parasites (template based uncatalyzed reactions) first order
        self.mu=mu #Vector of the mutation rates for replicases
        self.nu=nu #Vector of the mutation rates for parasites
        #if (self.mu[-1]!=0. or self.nu[-1]!=0):
        #    raise Exception("The final mutation rate should be 0.")
        self.mmutantmax=mmutmax #Number of mutant types for replicases
        self.ymutantmax=ymutmax #Number of mutant types for parasites
        self.mmutant=np.zeros(mmutmax) #create the array of the frequencies of each mutant for replicases
        self.ymutant=np.zeros(ymutmax) #create the array of the frequencies of each mutant for parasites
        self.mmutant[0]=xini #Intitially we only have non-muted replicases
        self.ymutant[0]=1-xini #Intitially we only have non-muted parasites
        if len(begin)>=1:
            for i in range(mmutmax):
                self.mmutant[i] = begin[i]
            for i in range(ymutmax):
                self.ymutant[i] = begin[i+mmutmax]
        if mutantfirst:
            self.mmutant[0]=0. #Intitially we only have non-muted replicases
            self.ymutant[0]=0. #Intitially we only have non-muted parasites
            self.mmutant[1]=xini #Intitially we only have muted replicases
            self.ymutant[1]=1-xini #Intitially we only have muted parasites
        self.xini=xini
        self.dilu = dilu
        self.printinfo = printinfo
        self.history = history
        self.begin = begin
        
    def reinit(self):
        self.mmutant=np.zeros(self.mmutantmax)
        self.ymutant=np.zeros(self.ymutantmax)
        self.mmutant[0]=self.xini 
        self.ymutant[0]=1-self.xini
        if len(self.begin)>=1:
            for i in range(self.mmutantmax):
                self.mmutant[i] = self.begin[i]
            for i in range(self.ymutantmax):
                self.ymutant[i] = self.begin[i+self.mmutantmax]
        
    def compart(self): #Create the compartments from the pool (step 1 of the transient process), following the repartition in the pool
        k=0
        self.comp=[]
        while k<self.N: #k is the updated number of compartments, N the total number of compartments
            n=np.random.poisson(self.lamda) #Pick the total number in the compartment according to a Poisson distribution
            marr=np.zeros(len(self.mmutant)) #Array of the numbers of different mutants of replicases
            yarr=np.zeros(len(self.ymutant)) #Array of the numbers of different mutants of parasites
            repart=np.concatenate((np.array(self.mmutant), np.array(self.ymutant))) #Fractions of each mutants (first the replicases then the parasites)
            for j in range(n): #Attribute the n available slots according to the fraction of each species in the pool
                p = np.random.rand() #Pick a random number between 0 and 1
                q=0
                l=0
                while q<p:
                    q+=repart[l] #Add the fraction of each species
                    l+=1
                l-=1
                if l<len(self.mmutant):
                    marr[l]+=1
                else:
                    yarr[l-len(self.mmutant)]+=1
            self.comp.append([marr,yarr,[-1.,0.],sum(marr)+sum(yarr),marr.copy(),yarr.copy(),0]) #self.comp contains a list with 2 arrays for each of the N compartments, 1 for the mutants of replicases, 1 for the mutants of parasites. [-1,0] represents the internal clock (previous and current times) of the compartment (initially), we store the initial distribution and the number of steps
            k+=1
        

    def step(self, comp, carryingcap = False): #Maturation of one of the different compartments (Gillespiebet simulation of the parallel evolutions in each compartment)
        S = 0.
        marr,yarr,tintm1,tint,nini=comp[0].copy(),comp[1].copy(),comp[2][0],comp[2][1],comp[3] #Initialize with the number of individuals for every mutation of parasites and replicases depending on the initial repartition in the compartment
        if not(carryingcap): #If the compartment has not reached the carrying capacity yet
            for i in range(self.mu.size):
                for j in range(self.mu.size):
                    S += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
            for i in range(self.mu.size):
                for j in range(self.nu.size):
                    S += ( self.tau[j] + marr[i] * self.gamma[i,j] ) * yarr[j] #replication of parasite using replicase
        for i in range(self.mu.size):
            S+= marr[i] * self.mu[i] #mutation of a replicase
        for j in range(self.nu.size):
            S+= marr[j] * self.nu[j] #mutation of a parasite
        x1 = np.random.random()
        x2 = np.random.random()
        tau = np.log(1 / x1) / S #time until next reaction
        R = 0.
        i = 0
        j = 0
        imut = 0
        jmut = 0
        mut=False
        iplus, jplus = False, False
        while R < x2 * S:
            if not(carryingcap):
                if ((i < self.mu.size - 1) and (j <= self.mu.size - 1)):
                    R += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
                    i += 1
                    iplus, jplus = True, False
                elif ((i == self.mu.size - 1) and (j <= self.mu.size - 1)):
                    R += ( self.kappa[j] + marr[i] * self.alpha[i,j] ) * marr[j] #replication of replicase
                    i = 0
                    j += 1
                    iplus, jplus = False, True
                elif ((i < self.mu.size - 1) and (j > self.mu.size - 1) and (j < self.mu.size + self.nu.size)):
                    jbis = j - self.mu.size
                    R += ( self.tau[jbis] + marr[i] * self.gamma[i,jbis] ) * yarr[jbis] #replication of parasite using replicase
                    i += 1
                    iplus, jplus = True, False
                elif ((i == self.mu.size - 1) and (j > self.mu.size - 1) and (j < self.mu.size + self.nu.size)):
                    jbis = j - self.mu.size
                    R += ( self.tau[jbis] + marr[i] * self.gamma[i,jbis] ) * yarr[jbis] #replication of parasite using replicase
                    i = 0
                    j += 1
                    iplus, jplus = False, True
                else: #A mutation occurs
                    mut=True
                    if imut <= self.mu.size -1:
                        R += marr[imut] * self.mu[imut] #mutation of a replicase
                        imut += 1
                        imutplus, jmutplus = True, False
                    else:
                        R += marr[jmut] * self.nu[jmut] #mutation of a parasite
                        jmut += 1
                        imutplus, jmutplus = False, True
            else : # If the compartment has reached the carrying capacity, only mutations can occur
                mut=True
                if imut <= self.mu.size -1:
                    R += marr[imut] * self.mu[imut] #mutation of a replicase
                    imut += 1
                    imutplus, jmutplus = True, False
                else:
                    R += marr[jmut] * self.nu[jmut] #mutation of a parasite
                    jmut += 1
                    imutplus, jmutplus = False, True
        comp[2][0] = tint
        tint += tau
        comp[2][1] = tint
        if mut:
            return [i-iplus,j-jplus,imut-imutplus,jmut-jmutplus,mut] #if a replicase replicates, j<self.mu.size and mut=False, if a parasite replicates, j>=self.mu.size and mut=False, if a replicase mutates, jmut=0 and mut=True, if a parasite mutates imut=self.mu.size - 1
        else:
            return [i-iplus,j-jplus,imut,jmut,mut]
        
    def allstep(self): #can be improved with Michele's method probably, perform the Gillespiebet step in each compartment
        Sm0, Sm1, Sy0, Sy1 = 0., 0., 0., 0.
        for p, comp in enumerate(self.comp):
            marr,yarr,tintm1,tint,nini=comp[0].copy(),comp[1].copy(),comp[2][0],comp[2][1],comp[3]
            if sum(comp[0])>0: #If the compartment is not empty
                carryingcap = ((sum(comp[0])+sum(comp[1])/self.ratiolen)>=self.K+nini)
                if (((sum(comp[0])+sum(comp[1])/self.ratiolen)<self.K+nini) and (tint<self.T)): #If the compartment has not reached its carrying capacity yet and the maturation time has not been exceeded
                    comp[-1] += 1
                    res=self.step(comp, carryingcap)
                    i,j,imut,jmut,mut = res[0], res[1], res[2], res[3], res[4]
                    ##Update the populations##
                    if comp[2][1]<self.T: #If the process has time to occur
                        if mut: #If a mutation has been picked
                            if (imut >= self.mu.size): #If a host has muted to parasite
                                if ((comp[4][1]==0 or comp[5][1]==0) and comp[4][0]>0 and comp[5][0]>0):
                                    self.ymutcount[0] += 1
                                else:
                                    self.ymutcount[1] += 1
                                marr[jmut]-=1 #The number of hosts is decreased by 1
                                yarr[jmut]+=1 #The number of mutant parasites is increased by 1
                            else : #If a replicase has muted
                                if (comp[4][1]==0 and comp[4][0]>0 and comp[5][0]==0):
                                    self.mmutcount[0] += 1
                                else:
                                    self.mmutcount[1] += 1
                                marr[imut+1]+=1 #The number of next type of mutant is increased by 1
                                marr[imut]-=1 #The number of next type of mutant is decreased by 1
                        else: #If a replication has been picked
                            jbis = j - self.mu.size
                            if  (j > self.mu.size - 1):
                                yarr[jbis] += 1 #Increase the population of parasites by 1 (doesn't depend on the replicase i used)
                            else:
                                marr[j] += 1 #Increase the population of replicases by 1 (doesn't depend on the replicase i used)
                    else:
                        comp[2][1]=self.T
                elif (tint<self.T): #If the carrying capacity has been reached but not the maturation time, mutations can still occur 
                    comp[-1] += 1
                    res=self.step(comp, carryingcap)
                    i,j,imut,jmut,mut = res[0], res[1], res[2], res[3], res[4]
                    ##Update the populations##
                    #We know that a mutation has been picked
                    if comp[2][1]<self.T:
                        if (imut >= self.mu.size): #If a parasite has muted
                            if ((comp[4][1]==0 or comp[5][1]==0) and comp[4][0]>0 and comp[5][0]>0):
                                self.ymutcount[0] += 1
                            else:
                                self.ymutcount[1] += 1
                            yarr[jmut]+=1 #The number of mutant parasites is increased by 1
                            marr[jmut]-=1 #The number of corresponding hosts is decreased by 1
                        else : #If a replicase has muted
                            if (comp[4][1]==0 and comp[4][0]>0 and comp[5][0]==0):
                                self.mmutcount[0] += 1
                            else:
                                self.mmutcount[1] += 1
                            marr[imut+1]+=1 #The number of next type of mutant is increased by 1
                            marr[imut]-=1 #The number of next type of mutant is decreased by 1
                    else:
                        comp[2][1]=self.T
            comp[0]=marr
            comp[1]=yarr
            Sm0 += marr[0]
            Sm1 += marr[1]
            Sy0 += yarr[0]
            Sy1 += yarr[1]
            self.comp[p]=comp
        self.mmutantnum,self.ymutantnum = [Sm0, Sm1], [Sy0, Sy1]
        
    def pooling(self): #After the maturation step, perform the pooling again by emptying the compartments in a common pool
        self.mmutant=np.zeros(self.mu.size) #initiate the new fractions of replicases
        self.ymutant=np.zeros(self.nu.size) #initiate the new fractions of parasites
        ntot=0 #Total number of individuals in the pool 
        for comp in self.comp: #Empty each compartment one by one
            for i in range(comp[0].size):
                ntot += comp[0][i] #Increase the total number of individuals in the pool
                self.mmutant[i] += comp[0][i] #Update the number of mutants of type i (replicases) in the pool
            for j in range(comp[1].size):
                ntot += comp[1][j] #Increase the total number of individuals in the pool
                self.ymutant[j] += comp[1][j] #Update the number of mutants of type i (parasites) in the pool
        self.lamda = (sum(self.mmutant)+sum(self.ymutant))/self.N/self.dilu
        self.mmutant = [self.mmutant[i]/ntot for i in range(comp[0].size)] #Creates the fractions of each mutant of replicases
        self.ymutant = [self.ymutant[i]/ntot for i in range(comp[1].size)] #Creates the fractions of each mutant of parasites

    def evolv(self, fraction = False, lamcond=False, xcond=False, carryingcaplim = True):
        self.reinit()
        repart = []
        lam=[self.lamda]
        x=[sum(self.mmutant)]
        x0ano, x1ano, z0ano, z1ano = 0, 0, 0, 0
        ntot = 0
        for k in range(self.repet): #Repeat the process repet times
            self.mmutcount = [0,0]
            self.ymutcount = [0,0]
            comp_hist=[[] for i in range(int(self.N))]
            if (sum(self.ymutant)<1.): #If there still are replicases
                self.compart()
                t=0
                condition_end=False
                while not(condition_end):
                    ntot = 0
                    self.allstep()
                    condition_end=True
                    tlist=[]
                    for i,comp in enumerate(self.comp):
                        if self.history:
                            comp_hist[i].append([comp[0],comp[1]])
                        tintm1, tint, nini=comp[2][0], comp[2][1], comp[3]
                        tlist.append(tint)
                        ntot += 1.
                        if carryingcaplim : #If the evolution is limited by the carrying capacity
                            if ((sum(comp[0])+sum(comp[1])/self.ratiolen)<self.K+nini and sum(comp[0])>0 and tint<self.T):
                                former_cond = condition_end
                                condition_end = False
                        else : #If the evolution is a limited by a time of maturation
                            if ((tint<self.T) and sum(comp[0])>0 and (tint != tintm1)):
                                condition_end = False
            self.pooling()
            if fraction:
                repart.append([self.mmutant,self.ymutant])
            else:
                repart.append([self.mmutantnum,self.ymutantnum])
            lam.append(self.lamda)
            x.append(sum(self.mmutant))
        if lamcond:
            if xcond:
                return [repart,x,lam]
            else:
                return [repart,lam]
        else:
            if xcond:
                return [repart,x]
            else:
                return repart
            
### Theoretical description WITH STIRRING ###
class theory_compart_stir_betedge:
    def __init__(self, T, K, d, x0ini, x1ini, z0ini, lamdaini, repet, alpha, gamma, mu, nu, ratiolen=4, tradeoff = 0, s=1., ncomp = int(1E3), randomstir = False, history=False, r1=0.25, r2=0.25, r3=0.25, r4=0.25):
        """
        Class for the theoretical description of transient compartmentalization dynamics with mutations and selection, within a Gillespie framework with stirring (Model B with trade offs)
        T : maturation time
        K : carrying capacity
        d : dilution factor after pooling
        x0ini : initial fraction of non-mutated replicases in the pool
        x1ini : initial fraction of mutated replicases in the pool
        z0ini : initial fraction of non-mutated parasites in the pool
        lamdaini : initial parameter of the poisson distribution for the number of individuals in each compartment
        repet : number of repetitions of the whole process
        alpha : matrix of the reproduction rates for replicases (template based catalyzed reactions) second order
        gamma : matrix of the reproduction rates for parasites (template based catalyzed reactions) second order
        mu : vector of the mutation rates for replicases
        nu : vector of the mutation rates for parasites
        ratiolen : ratio of lengths between parasites and replicases (default 4)
        tradeoff : if 0 no trade off, if 1 linear trade off, if 2 stepwise trade off (default 0)
        s : fraction of the population that is stirred at each round (default 1.)
        ncomp : number of compartments
        randomstir : if True, the fraction of the population that is stirred at each round is random between 0.98 and 1 (default False)
        history : if True, the history of each compartment at each step is stored (default False)
        r1, r2, r3, r4 : ratios of the carrying capacities of different species (default 0.25 for each)  
        """
        self.T = T #In hours
        self.K = K
        self.d = d
        self.ratiolen = ratiolen
        self.mmutant = np.array([x0ini,x1ini])
        self.ymutant = np.array([z0ini,1-x0ini-x1ini-z0ini])
        self.x0 = x0ini
        self.x1 = x1ini
        self.z0 = z0ini
        self.z1 = 1-x0ini-x1ini-z0ini
        self.lamda = lamdaini
        self.repet = repet
        self.mu = mu #In 1/h
        self.nu = nu #In 1/h
        self.gamma = gamma #In 1/h
        self.alpha = alpha #In 1/h
        self.fastmut = False
        self.ncomp = ncomp
        self.tradeoff = tradeoff
        self.s = s
        self.ncomp = ncomp
        self.randomstir = randomstir
        self.history = history
        self.r1, self.r2, self.r3, self.r4 = r1, r2, r3, r4

    def fact(self, n): #Factorial function for Poisson distribution
        pi, k = 1, 0
        while k < n:
            k+=1
            pi*=k
        return pi
        
    def initialization(self):
        self.comp, lam = [], self.lamda
        x0, x1, z0, z1 = self.x0, self.x1, self.z0, self.z1
        fracs = [x0, x1, z0, z1]
        Sm0, Sm1, Sy0, Sy1 = 0., 0., 0., 0.
        for i in range(self.ncomp):
            r = np.random.rand()
            poisson, n = 0., 0
            while poisson < r:
                poisson += np.exp(-lam)*(lam**n)/(self.fact(n))
                n+=1
            n-=1
            compart = [0., 0., 0., 0.]
            for j in range(n):
                r, p = np.random.rand(), 0.
                for k, frac in enumerate(fracs):
                    p += frac
                    if r<=p:
                        break
                compart[k] += 1
            self.comp.append(compart)
            Sm0, Sm1, Sy0, Sy1 = Sm0 + compart[0], Sm1 + compart[1], Sy0 + compart[2], Sy1 + compart[3]
            
    def update(self):
        x0, x1, z0, z1, K, d, lam, s, ncomp = self.x0, self.x1, self.z0, self.z1, self.K, self.d, self.lamda, self.s, self.ncomp
        if self.randomstir:
            s = np.random.rand()*(1.-0.98) + 0.98
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        Sm0, Sm1, Sy0, Sy1 = 0., 0., 0., 0.
        ## Evolve the compartments ##
        if self.history:
            hist_compbycomp_step=[]
        for i, compart in enumerate(self.comp):
            m0i, m1i, y0i, y1i, ni = compart[0], compart[1], compart[2], compart[3], sum(compart)
            if self.tradeoff == 2:
                if (m1i < 1) and (y0i < 1) and (m0i > 0) and (y1i < 1):
                    compart[0] += K
                elif (m1i < 1) and (y0i < 1) and (m0i > 0)  and (y1i > 0):
                    compart[0] += K/2
                    compart[3] += K*self.ratiolen/2
                elif (y0i > 0) and (m0i > 0):
                    compart[2] += K*self.ratiolen
                elif (m1i > 0) and (y0i < 1) and (m0i > 0)  and (y1i < 1):
                    compart[0] += K/2
                    compart[1] += K/2
                elif (m1i > 0) and (m0i < 1)  and (y1i < 1):
                    compart[1] += K
                elif (m1i > 0) and (y0i < 1) and (m0i < 1)  and (y1i > 0):
                    compart[1] += K/2
                    compart[3] += K*self.ratiolen/2
                elif (m1i > 0) and (y0i < 1) and (m0i > 0) and (y1i > 0):
                    compart[0] += K/3
                    compart[1] += K/3
                    compart[3] += K*self.ratiolen/3
                if self.history:
                    hist_compbycomp_step.append([compart[0], compart[1], compart[2], compart[3]]) 
                #mutations
                if (m1i < 1) and (y0i < 1) and (m0i > 0) and (y0i < 1):
                    mut = self.mu[0]/self.alpha[0,0] * np.log(K+1) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                    mut = self.nu[0]/self.alpha[0,0] * np.log(K+1) + self.nu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m0i)
                    compart[0] -= mut
                    compart[2] += mut
                elif (m1i < 1) and (y0i < 1) and (m0i > 0)  and (y1i > 0):
                    mut = self.mu[0]/self.alpha[0,0] * np.log(K/2+1) + self.mu[0]*max(0,T-K/2/(self.alpha[0,0]*(K/2+1)))*(K/2+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                    mut = self.nu[0]/self.alpha[0,0] * np.log(K/2+1) + self.nu[0]*max(0,T-K/2/(self.alpha[0,0]*(K/2+1)))*(K/2+m0i)
                    compart[0] -= mut
                    compart[2] += mut
                elif (y0i > 0) and (m0i > 0):
                    mut = T*self.mu[0]*m0i
                    mut0 = T*self.nu[0]*m0i
                    mut1 = T*self.nu[1]*m1i
                    compart[0] -= mut + mut0
                    compart[1] += mut
                    compart[2] += mut0
                    compart[1] -= mut1
                    compart[3] += mut1
                elif (m1i > 0) and (y0i < 1) and (m0i > 0)  and (y1i < 1):
                    mut = self.mu[0]/self.alpha[0,0] * np.log(K/2+1) + self.mu[0]*max(0,T-K/2/(self.alpha[0,0]*(K/2+1)))*(K/2+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                    mut = self.nu[0]/self.alpha[0,0] * np.log(K/2+1) + self.nu[0]*max(0,T-K/2/(self.alpha[0,0]*(K/2+1)))*(K/2+m0i)
                    compart[0] -= mut
                    compart[2] += mut
                    mut1 = self.nu[1]/self.alpha[0,0] * np.log(K/2+1) + self.nu[1]*max(0,T-K/2/(self.alpha[0,0]*(K/2+1)))*(K/2+m1i)
                    compart[1] -= mut1
                    compart[3] += mut1
                elif (m1i > 0) and (m0i < 1)  and (y1i < 1):
                    mut1 = self.nu[1]/self.alpha[0,0] * np.log(K+1) + self.nu[1]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m1i)
                    compart[1] -= mut1
                    compart[3] += mut1
                elif (m1i > 0) and (y0i < 1) and (m0i < 1)  and (y1i > 0):
                    mut1 = self.nu[1]/self.alpha[0,0] * np.log(K/2+1) + self.nu[1]*max(0,T-K/2/(self.alpha[0,0]*(K/2+1)))*(K/2+m1i)
                    compart[1] -= mut1
                    compart[3] += mut1
                elif (m1i > 0) and (y0i < 1) and (m0i > 0) and (y1i > 0):
                    mut = self.mu[0]/self.alpha[0,0] * np.log(K/3+1) + self.mu[0]*max(0,T-K/3/(self.alpha[0,0]*(K/3+1)))*(K/3+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                    mut = self.nu[0]/self.alpha[0,0] * np.log(K/3+1) + self.nu[0]*max(0,T-K/3/(self.alpha[0,0]*(K/3+1)))*(K/3+m0i)
                    compart[0] -= mut
                    compart[2] += mut
                    mut1 = self.nu[1]/self.alpha[0,0] * np.log(K/3+1) + self.nu[1]*max(0,T-K/3/(self.alpha[0,0]*(K/3+1)))*(K/3+m1i)
                    compart[1] -= mut1
                    compart[3] += mut1
            elif self.tradeoff == 1:
                if (m1i < 1) and (y0i < 1) and (m0i > 0) and (y0i <1):
                    compart[0] += K
                elif (m1i < 1) and (y0i < 1) and (m0i > 0)  and (y1i >0):
                    compart[0] += K/2
                    compart[3] += K/2
                elif (m1i > 0) and (y1i < 1) and ((m0i < 1) or (y0i < 1)):
                    compart[1] += K
                elif (m1i > 0) and (y1i > 0) and ((m0i < 1) or (y0i < 1)):
                    compart[1] += K/2
                    compart[3] += K/2
                elif (m0i > 0) and (y0i > 0):
                    compart[2] += K
                if self.history:
                    hist_compbycomp_step.append([compart[0], compart[1], compart[2], compart[3]])
                #mutations
                if (m1i < 1) and (y0i < 1) and (m0i > 0) and (y1i < 1): #if m0 is favoured
                    mut = self.mu[0]/self.alpha[0,0] * np.log(K+1) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                    mut = self.nu[0]/self.alpha[0,0] * np.log(K+1) + self.nu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m0i)
                    compart[0] -= mut
                    compart[2] += mut
                elif (m1i > 0) and (y0i < 1) and (m0i > 0) and (y1i < 1): #if m0 and m1 are favoured
                    mut = self.mu[0]/self.alpha[0,0] * np.log(K/2) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K/2+m0i)
                    mut1 = self.nu[1]/self.alpha[0,0] * np.log(K/2) + self.nu[1]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K/2+m1i)
                    compart[0] -= mut
                    compart[1] += mut
                    compart[1] -= mut1
                    compart[3] += mut1
                    mut = self.nu[0]/self.alpha[0,0] * np.log(K/2) + self.nu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K/2+m0i)
                    compart[0] -= mut
                    compart[2] += mut
                elif (m1i > 0) and (m0i < 1) and (y1i < 1): #if m1 is favoured
                    mut1 = self.nu[1]/self.alpha[0,0] * np.log(K+1) + self.nu[1]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m1i)
                    compart[1] -= mut1
                    compart[3] += mut1
                else: #if none replicases are favoured
                    mut = T*self.mu[0]*m0i
                    mut0 = T*self.nu[0]*m0i
                    mut1 = T*self.nu[1]*m1i
                    compart[0] -= mut + mut0
                    compart[1] += mut
                    compart[2] += mut0
                    compart[1] -= mut1
                    compart[3] += mut1         
            elif self.tradeoff == 4:
                if (m0i > 0) and (m1i < 1) and (y1i < 1):
                    compart[0] += K
                elif (m0i < 1)  and (m1i > 0) and (y0i < 1) and (y1i < 1):
                    compart[1] += K
                elif (m0i > 0) and (m1i > 0) and (y1i < 1) and (y0i < 1):
                    compart[0] += K*self.r1
                    compart[1] += K*(1-self.r1)
                elif (m1i > 0) and (y0i > 0):
                    compart[2] += K*self.ratiolen
                elif (m0i > 0) and (m1i < 1) and (y0i < 1) and (y1i > 0):
                    compart[0] += K*self.r2
                    compart[3] += K*self.ratiolen*(1-self.r2)
                elif (m0i < 1) and (m1i > 0) and (y0i < 1) and (y1i > 0):
                    compart[1] += K*self.r3
                    compart[3] += K*self.ratiolen*(1-self.r3)
                elif (m0i > 0) and (m1i > 0) and (y0i < 1) and (y1i > 0):
                    compart[0] += K*self.r1*self.r4
                    compart[1] += K*(1-self.r1)*self.r4
                    compart[3] += K*self.ratiolen*(1-self.r4)
                #mutations
                if (m0i > 0) and (m1i < 1) and (y0i <1):
                    mut = self.mu[0]/self.alpha[0,0] * np.log(K+1) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                elif (m0i < 1)  and (m1i > 0) and (y0i < 1) and (y1i < 1):
                    mut = self.nu[0]/self.alpha[0,0] * np.log(K+1) + self.nu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(K+m1i)
                    compart[0] -= mut
                    compart[1] += mut
                elif (m0i > 0) and (m1i > 0) and (y1i < 1) and (y0i < 1):
                    mut = self.mu[0]/self.alpha[0,0] * np.log(self.r1*K+1) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(self.r1*K+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                    mut = self.nu[0]/self.alpha[0,0] * np.log((1-self.r1)*K+1) + self.nu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*((1-self.r1)*K+m1i)
                    compart[1] -= mut
                    compart[2] += mut
                elif (m1i > 0) and (y0i > 0):
                    mut = self.nu[1]/self.alpha[0,0] * np.log(self.ratiolen*K+1) + self.nu[1]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(self.ratiolen*K+y0i)
                    compart[2] -= mut
                    compart[3] += mut
                    mut = T*self.mu[0]*m0i
                    mut0 = T*self.nu[0]*m1i
                    compart[0] -= mut
                    compart[1] += mut
                    compart[1] -= mut0
                    compart[2] += mut0
                elif (m0i > 0) and (y0i < 1) and (m1i < 1) and (y1i > 0):
                    mut = self.mu[0]/self.alpha[0,0] * np.log(self.r2*K+1) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(self.r2*K+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                elif (m0i < 1)  and (m1i > 0) and (y0i < 1) and (y1i > 0):
                    mut = self.nu[0]/self.alpha[0,0] * np.log(self.r3*K+1) + self.nu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(self.r3*K+m1i)
                    compart[1] -= mut
                    compart[2] += mut
                elif (m0i > 0) and (m1i > 0) and  (y0i < 1) and (y1i > 0):
                    mut = self.mu[0]/self.alpha[0,0] * np.log(self.r1*self.r4*K+1) + self.mu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*(self.r1*self.r4*K+m0i)
                    compart[0] -= mut
                    compart[1] += mut
                    mut = self.nu[0]/self.alpha[0,0] * np.log((1-self.r1)*self.r4*K+1) + self.nu[0]*max(0,T-K/(self.alpha[0,0]*(K+1)))*((1-self.r1)*self.r4*K+m1i)
                    compart[1] -= mut
                    compart[2] += mut
            Sm0, Sm1, Sy0, Sy1 = Sm0 + compart[0], Sm1 + compart[1], Sy0 + compart[2], Sy1 + compart[3]
            self.comp[i] = compart
        if self.history:
            self.hist_compbycomp.append(hist_compbycomp_step)
        Sn = Sm0 + Sm1 + Sy0 + Sy1
        x0, x1, z0, z1 = Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0)), Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))
        self.mmutantnum, self.ymutantnum = np.array([Sm0, Sm1]), np.array([Sy0, Sy1])
        ## Dilute ##
        Ndil = ncomp//d
        remove = rand.sample(range(ncomp),int(Ndil))
        for k in remove:
            self.comp[k] = [0., 0., 0., 0.]
        self.lamda = Sn/d/ncomp
        ## Stirring ##
        Sm0, Sm1, Sy0, Sy1 = 0, 0, 0, 0
        for i, compart in enumerate(self.comp):
            m0i, m1i, y0i, y1i, ni = compart[0], compart[1], compart[2], compart[3], sum(compart)
            m0i, m1i, y0i, y1i = (1-s)*m0i + s*x0*Sn/d/ncomp, (1-s)*m1i + s*x1*Sn/d/ncomp, (1-s)*y0i + s*z0*Sn/d/ncomp, (1-s)*y1i + s*z1*Sn/d/ncomp
            lami = (1-s)*ni + s*self.lamda
            lami += (lami==0.)
            x0i, x1i, z0i, z1i = m0i/lami, m1i/lami, y0i/lami, y1i/lami
            fracs = [x0i, x1i, z0i, z1i]
            r = np.random.rand()
            poisson, n = 0., 0
            while poisson < r:
                poisson += np.exp(-lami)*(lami**n)/(self.fact(n))
                n+=1
            n-=1
            compart = [0., 0., 0., 0.]
            for j in range(n):
                r, p = np.random.rand(), 0.
                for k, frac in enumerate(fracs):
                    p+=frac
                    if r<=p:
                        break
                compart[k] += 1
            Sm0, Sm1, Sy0, Sy1 = Sm0 + compart[0], Sm1 + compart[1], Sy0 + compart[2], Sy1 + compart[3]
            self.comp[i] = compart
        Sn = Sm0 + Sm1 + Sy0 + Sy1
        self.x0, self.x1, self.z0, self.z1 = Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0)), Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))
        self.mmutant, self.ymutant = np.array([Sm0/(Sn + (Sn==0)), Sm1/(Sn + (Sn==0))]), np.array([Sy0/(Sn + (Sn==0)), Sy1/(Sn + (Sn==0))])
        
    def evol(self, fraction = False):
        self.initialization()
        if self.history:
            self.hist_compbycomp=[]
        hist_evol = []
        for k in range(self.repet):
            self.update()
            if fraction:
                hist_evol.append([self.mmutant,self.ymutant])
            else:
                hist_evol.append([self.mmutantnum,self.ymutantnum])   
        if self.history:
            return np.array(hist_evol), np.array(self.hist_compbycomp)
        else:
            return np.array(hist_evol)

### Theoretical description ###

class theory_compart_bet:
    def __init__(self, T, K, d, x0ini, x1ini, z0ini, lamdaini, repet, alpha, gamma, mu, nu, r1=0.5, r2=0.5, r3=0.5, r4=0.5, ratiolen=4, tradeoff = 0, ncomp = 1E3):
        """
        """
        self.T = T
        self.K = K
        self.d = d
        self.ratiolen = ratiolen
        self.mmutant = np.array([x0ini,x1ini])
        self.ymutant = np.array([z0ini,1-x0ini-x1ini-z0ini])
        self.x0 = x0ini
        self.x1 = x1ini
        self.z0 = z0ini
        self.z1 = 1-x0ini-x1ini-z0ini
        self.m0, self.m1, self.y0, self.y1 = lamdaini*x0ini, lamdaini*x1ini, lamdaini*z0ini, lamdaini*self.z1,
        self.lamda = lamdaini
        self.repet = repet
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.alpha = alpha
        self.ratio1 = r1
        self.ratio2 = r2
        self.ratio3 = r3
        self.ratio4 = r4
        self.smartmut = False
        if (self.gamma[0][1] > 0.) and (tradeoff == 0):
            ##print('smartmut = ', self.smartmut)
            self.smartmut = True
        self.ncomp = ncomp
        self.tradeoff = tradeoff
    
    def x0star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        r1 = self.ratio1
        r2 = self.ratio2
        r3 = self.ratio3
        r4 = self.ratio4
        #growth
        if self.tradeoff == 0: #Resistance/replicability
            nstar = lam + K*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) + K*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.)
            m0starnomut = K*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)-1.) + lam * x0
        elif self.tradeoff == 1: #Generality/Replicability
            nstar = lam + K*np.exp(-lam)*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z1)-1.) + K*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + (K/2)*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2) * np.exp(-lam) * (np.exp(lam*z1) - 1) * (np.exp(lam*x1)*(np.exp(lam*x0)-1.) + np.exp(lam*z0)*(np.exp(lam*x1)-1.))
            m0starnomut = K*np.exp(-lam)*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z1)-1.) + lam * x0
        elif self.tradeoff == 2: #Generality/Replicability (slow mutants)
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * (1. + 1/2 * ( (np.exp(lam* x1) - 1.) + (np.exp(lam* z1) - 1.) + 2/3 * (np.exp(lam* x1) - 1.) * (np.exp(lam* z1) - 1.) ) ) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( np.exp(lam*z0) * (1 + 1/2 *(np.exp(lam*z1) - 1.)) + 1/2 * (np.exp(lam*x0) - 1.) + 1/3 * (np.exp(lam*x0) - 1.) * (np.exp(lam*z1) - 1.)) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1 + z1)) * (np.exp(lam * x0) - 1.) * (np.exp(lam * z0) - 1.) + K*self.ratiolen/2 * np.exp(-lam) * ((np.exp(lam * x0) - 1.) * (np.exp(lam * z1) - 1.) + np.exp(lam * z0)*(np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.) + 2/3 * (np.exp(lam * x0) - 1.) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.))
            m0starnomut = K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * (1. + 1/2 * ( (np.exp(lam* x1) - 1.) + (np.exp(lam* z1) - 1.) + 2/3 * (np.exp(lam* x1) - 1.) * (np.exp(lam* z1) - 1.) ) ) + lam * x0
        else: #General for exp
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * ( np.exp(lam * z0) + r1 * (np.exp(lam * x1) - 1.) + r2 * (np.exp(lam * z1) - 1) + r1 * r4 * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( 1 + (1 - r1) * (np.exp(lam * x0) - 1) + r3 * (np.exp(lam * z1) - 1) + (1- r1) * r4 * (np.exp(lam * x0) - 1) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x0 + z1)) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z0) - 1.) + K * np.exp(-lam)*self.ratiolen * (np.exp(lam * z1) - 1) * ((1 - r2) * (np.exp(lam * x0) - 1) + (1 - r3) * (np.exp(lam * x1) - 1) + (1 - r4) * (np.exp(lam * x0) - 1) * (np.exp(lam * x1) - 1) )
            m0starnomut = K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * ( np.exp(lam * z0) + r1 * (np.exp(lam * x1) - 1.) + r2 * (np.exp(lam * z1) - 1) + r1 * r4 * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1)) + lam * x0
        #mutations
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        if self.tradeoff ==0:
            m0starmutfav = -( ( ((self.mu[0]+self.nu[0])/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)  + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*(x0+z1))) 
            m0starmutunfav = -self.T*(self.mu[0]+self.nu[0])*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*(x1+z0))-1.) 
        elif self.tradeoff ==1:
            m0starmutfav = - ( ( ((self.mu[0]+self.nu[0])/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)  + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0))
            m0starmutunfav = - self.T*(self.mu[0]+self.nu[0])*lam*x0*np.exp(-lam)*np.exp(lam*x0)*(np.exp(lam*(x1+z0+z1))-1.)
        elif self.tradeoff == 2:
            m0starmutfav = - ( ( ((self.mu[0]+self.nu[0])/self.alpha[0][0])*np.log(K) + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.) + ( ((self.mu[0]+self.nu[0])/self.alpha[0][0])*np.log(K/2) + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*K/2) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.) + ( ((self.mu[0]+self.nu[0])/self.alpha[0][0])*np.log(K/3) + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*K/3) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1.)  + ((self.mu[0]+self.nu[0])*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0))
            m0starmutunfav = - self.T*(self.mu[0]+self.nu[0])*lam*x0*np.exp(-lam)*np.exp(lam*(x0+x1+z1))*(np.exp(lam*z0)-1.)
            ##print(m0starmutfav,m0starmutunfav)
        else :
            m0starmutfav = - ( ( (self.mu[0]/self.alpha[0][0])*np.log(K) + (self.mu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*np.exp(lam*z0) + ( (self.mu[0]/self.alpha[0][0])*np.log(r1*K) + (self.mu[0]*max(0,(self.T - Tfin))*r1*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1) + ( (self.mu[0]/self.alpha[0][0])*np.log(r2*K) + (self.mu[0]*max(0,(self.T - Tfin))*r2*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1) + ( (self.mu[0]/self.alpha[0][0])*np.log(r1*r4*K) + (self.mu[0]*max(0,(self.T - Tfin))*r1*r4*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1))
            m0starmutunfav = - self.T*self.mu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*x1)-1.)*(np.exp(lam*z0)-1.)
        if x0 == 0. or self.mu[0]==0.: #Case when there is no mutations of replicases
            m0starmutfav = 0.
            m0starmutunfav = 0.
        value = max((m0starnomut + m0starmutfav + m0starmutunfav)/nstar,0.)
        self.m0 = m0starnomut + m0starmutfav + m0starmutunfav
        ##print('m0 ', self.m0)
        return value
    
    def x1star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        self.n = lam
        r1 = self.ratio1
        r2 = self.ratio2
        r3 = self.ratio3
        r4 = self.ratio4
        #nstar = lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        if self.tradeoff == 0:
            nstar = lam + K*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) + K*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.)
            m1starnomut = lam*x1 + (K/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) + K*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)
        elif self.tradeoff == 1:
            nstar = lam + K*np.exp(-lam)*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z1)-1.) + K*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + (K/2)*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2) * np.exp(-lam) * (np.exp(lam*z1) - 1) * (np.exp(lam*x1)*(np.exp(lam*x0)-1.) + np.exp(lam*z0)*(np.exp(lam*x1)-1.))
            m1starnomut = lam * x1 + K*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + (K/2)*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1)
        elif self.tradeoff == 2:
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * (1. + 1/2 * ( (np.exp(lam* x1) - 1.) + (np.exp(lam* z1) - 1.) + 2/3 * (np.exp(lam* x1) - 1.) * (np.exp(lam* z1) - 1.) ) ) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( np.exp(lam*z0) * (1 + 1/2 *(np.exp(lam*z1) - 1.)) + 1/2 * (np.exp(lam*x0) - 1.) + 1/3 * (np.exp(lam*x0) - 1.) * (np.exp(lam*z1) - 1.)) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1 + z1)) * (np.exp(lam * x0) - 1.) * (np.exp(lam * z0) - 1.) + K*self.ratiolen/2 * np.exp(-lam) * ((np.exp(lam * x0) - 1.) * (np.exp(lam * z1) - 1.) + np.exp(lam * z0)*(np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.) + 2/3 * (np.exp(lam * x0) - 1.) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.))
            m1starnomut = K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( np.exp(lam*z0) * (1 + 1/2 *(np.exp(lam*z1) - 1.)) + 1/2 * (np.exp(lam*x0) - 1.) + 1/3 * (np.exp(lam*x0) - 1.) * (np.exp(lam*z1) - 1.)) + lam * x1
        else:
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * ( np.exp(lam * z0) + r1 * (np.exp(lam * x1) - 1.) + r2 * (np.exp(lam * z1) - 1) + r1 * r4 * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( 1 + (1 - r1) * (np.exp(lam * x0) - 1) + r3 * (np.exp(lam * z1) - 1) + (1- r1) * r4 * (np.exp(lam * x0) - 1) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x0 + z1)) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z0) - 1.) + K * np.exp(-lam)*self.ratiolen * (np.exp(lam * z1) - 1) * ((1 - r2) * (np.exp(lam * x0) - 1) + (1 - r3) * (np.exp(lam * x1) - 1) + (1 - r4) * (np.exp(lam * x0) - 1) * (np.exp(lam * x1) - 1) )
            m1starnomut = K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( 1 + (1 - r1) * (np.exp(lam * x0) - 1) + r3 * (np.exp(lam * z1) - 1) + (1- r1) * r4 * (np.exp(lam * x0) - 1) * (np.exp(lam * z1) - 1)) + lam * x1
            ##print('x1 before ', x1)
        #mutations
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        if self.tradeoff ==0:
            m1starmutfav = ( ( (self.mu[0]/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + (self.mu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)  + (self.mu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*(x0+z1))) - (self.nu[1]*np.log(1/(1-self.alpha[0][0]*Tfin))/self.alpha[1,1] + self.nu[1]*K*max(0,(self.T - Tfin)))*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0)+np.exp(lam*z0)-1.) - lam*x1*self.nu[1]*max(0,(self.T - Tfin))*np.exp(-lam)*np.exp(lam*x1)*(np.exp(lam*x0)+np.exp(lam*z0)-1.)
            m1starmutunfav = self.T*(self.mu[0])*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*(x1+z0))-1.) - lam*x1*self.nu[1]*self.T*np.exp(-lam)*np.exp(lam*x1)*((np.exp(lam*z1)-1.)*np.exp(lam*(x0+z0)) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.))
        elif self.tradeoff ==1:
            m1starmutfav = ( ( (self.mu[0]/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + (self.mu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)  + (self.mu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0)) - (self.nu[1]*np.log(1/(1-self.alpha[0][0]*Tfin))/self.alpha[1,1] + self.nu[1]*K*max(0,(self.T - Tfin)))*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0)+np.exp(lam*z0)-1.) - lam*x1*self.nu[1]*max(0,(self.T - Tfin))*np.exp(-lam)*np.exp(lam*x1)*(np.exp(lam*x0)+np.exp(lam*z0)-1.)
            m1starmutunfav = self.T*self.mu[0]*lam*x0*np.exp(-lam)*np.exp(lam*x0)*(np.exp(lam*(x1+z0+z1))-1.) - lam*x1*self.nu[1]*self.T*np.exp(-lam)*np.exp(lam*x1)*((np.exp(lam*z1)-1.)*np.exp(lam*(x0+z0)) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.))
        elif self.tradeoff == 2:
            m1starmutfav = ( ( (self.mu[0]/self.alpha[0][0])*np.log(K) + (self.mu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.) + ( (self.mu[0]/self.alpha[0][0])*np.log(K/2) + (self.mu[0]*max(0,(self.T - Tfin))*K/2) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.) + ( (self.mu[0]/self.alpha[0][0])*np.log(K/3) + (self.mu[0]*max(0,(self.T - Tfin))*K/3) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1.)  + (self.mu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0))
            m1starmutunfav = self.T*self.mu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+x1+z1))*(np.exp(lam*z0)-1.)
            m1starmutfav -= ( ( (self.nu[1]/self.alpha[1][1])*np.log(K) + (self.nu[1]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1)-1.) + ( (self.nu[1]/self.alpha[1][1])*np.log(K/2) + (self.nu[1]*max(0,(self.T - Tfin))*K/2) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.) + ( (self.nu[1]/self.alpha[1][1])*np.log(K/3) + (self.nu[1]*max(0,(self.T - Tfin))*K/3) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1.)  + (self.nu[1]*max(0,(self.T - Tfin))*lam*x1)*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)+np.exp(lam*z0)-1) )
            m1starmutunfav -= self.T*self.nu[1]*lam*x1*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)
            ##print(m1starmutfav,m1starmutunfav)
        else:
            m1starmutfav = ( ( (self.mu[0]/self.alpha[0][0])*np.log(K) + (self.mu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*np.exp(lam*z0) + ( (self.mu[0]/self.alpha[0][0])*np.log(r1*K) + (self.mu[0]*max(0,(self.T - Tfin))*r1*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1) + ( (self.mu[0]/self.alpha[0][0])*np.log(r2*K) + (self.mu[0]*max(0,(self.T - Tfin))*r2*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1) + ( (self.mu[0]/self.alpha[0][0])*np.log(r1*r4*K) + (self.mu[0]*max(0,(self.T - Tfin))*r1*r4*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1))
            m1starmutfav -= ( ( (self.nu[0]/self.alpha[0][0])*np.log(K) + (self.nu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x1)-1.) + ( (self.nu[0]/self.alpha[0][0])*np.log((1-r1)*K) + (self.nu[0]*max(0,(self.T - Tfin))*(1-r1)*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1) + ( (self.nu[0]/self.alpha[0][0])*np.log(r3*K) + (self.nu[0]*max(0,(self.T - Tfin))*r3*K) )*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1) + ( (self.nu[0]/self.alpha[0][0])*np.log((1-r1)*r4*K) + (self.nu[0]*max(0,(self.T - Tfin))*(1-r1)*r4*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1))
            m1starmutunfav = self.T*self.mu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*x1)-1.)*(np.exp(lam*z0)-1.)
            m1starmutunfav -= self.T*self.nu[0]*lam*x1*np.exp(-lam)*np.exp(lam*(x0+x1+z1))*(np.exp(lam*z0)-1.)
        if x0 == 0. or self.mu[0]==0.: #Case when there is no mutations of replicases
            m1starmutfav = 0.
            m1starmutunfav = 0.
        else:
            self.mmutcount[0] += m1starmutfav
            self.mmutcount[1] += m1starmutunfav
        value = max((m1starnomut + m1starmutfav + m1starmutunfav)/nstar,0.)
        self.m1 = m1starnomut + m1starmutfav + m1starmutunfav
        ##print('m1 ', self.m1)
        return value
    
    def z0star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        r1 = self.ratio1
        r2 = self.ratio2
        r3 = self.ratio3
        r4 = self.ratio4
        #nstar = lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        if self.tradeoff == 0:
            nstar = lam + K*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) + K*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.)
            y0starnomut = lam * z0 + K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)
        elif self.tradeoff == 1:
            nstar = lam + K*np.exp(-lam)*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z1)-1.) + K*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + (K/2)*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2) * np.exp(-lam) * (np.exp(lam*z1) - 1) * (np.exp(lam*x1)*(np.exp(lam*x0)-1.) + np.exp(lam*z0)*(np.exp(lam*x1)-1.))
            y0starnomut = lam * z0 + K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)
        elif self.tradeoff == 2:
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * (1. + 1/2 * ( (np.exp(lam* x1) - 1.) + (np.exp(lam* z1) - 1.) + 2/3 * (np.exp(lam* x1) - 1.) * (np.exp(lam* z1) - 1.) ) ) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( np.exp(lam*z0) * (1 + 1/2 *(np.exp(lam*z1) - 1.)) + 1/2 * (np.exp(lam*x0) - 1.) + 1/3 * (np.exp(lam*x0) - 1.) * (np.exp(lam*z1) - 1.)) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1 + z1)) * (np.exp(lam * x0) - 1.) * (np.exp(lam * z0) - 1.) + K*self.ratiolen/2 * np.exp(-lam) * ((np.exp(lam * x0) - 1.) * (np.exp(lam * z1) - 1.) + np.exp(lam * z0)*(np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.) + 2/3 * (np.exp(lam * x0) - 1.) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.))
            y0starnomut = K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x1 + z1)) * (np.exp(lam * x0) - 1.) * (np.exp(lam * z0) - 1.) + lam * z0
            ##print(y0starnomut)
        else:
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * ( np.exp(lam * z0) + r1 * (np.exp(lam * x1) - 1.) + r2 * (np.exp(lam * z1) - 1) + r1 * r4 * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( 1 + (1 - r1) * (np.exp(lam * x0) - 1) + r3 * (np.exp(lam * z1) - 1) + (1- r1) * r4 * (np.exp(lam * x0) - 1) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x0 + z1)) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z0) - 1.) + K * np.exp(-lam)*self.ratiolen * (np.exp(lam * z1) - 1) * ((1 - r2) * (np.exp(lam * x0) - 1) + (1 - r3) * (np.exp(lam * x1) - 1) + (1 - r4) * (np.exp(lam * x0) - 1) * (np.exp(lam * x1) - 1) )
            y0starnomut = K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x0 + z1)) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z0) - 1.) + lam * z0
        #mutations
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        if self.tradeoff ==0:
            y0starmutfav = ( ( (self.nu[0]/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + (self.nu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z1)*(np.exp(lam*x0)-1.)  + (self.nu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*(x0+z1))) 
            y0starmutunfav = self.T*self.nu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*(x1+z0))-1.) 
        elif self.tradeoff ==1:
            y0starmutfav = ( ( (self.nu[0]/self.alpha[0][0])*np.log(1/(1-self.alpha[0][0]*Tfin)) + (self.nu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)  + (self.nu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0))
            y0starmutunfav = self.T*self.nu[0]*lam*x0*np.exp(-lam)*np.exp(lam*x0)*(np.exp(lam*(x1+z0+z1))-1.)
        elif self.tradeoff == 2:
            y0starmutfav = ( ( (self.nu[0]/self.alpha[0][0])*np.log(K) + (self.nu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.) + ( (self.nu[0]/self.alpha[0][0])*np.log(K/2) + (self.nu[0]*max(0,(self.T - Tfin))*K/2) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.) + ( (self.nu[0]/self.alpha[0][0])*np.log(K/3) + (self.nu[0]*max(0,(self.T - Tfin))*K/3) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1.)  + (self.nu[0]*max(0,(self.T - Tfin))*lam*x0)*np.exp(-lam)*np.exp(lam*x0))
            y0starmutunfav = self.T*self.nu[0]*lam*x0*np.exp(-lam)*np.exp(lam*(x0+x1+z1))*(np.exp(lam*z0)-1.)
            ##print(y0starmutfav,y0starmutunfav)
        else:
            y0starmutfav = ( ( (self.nu[0]/self.alpha[0][0])*np.log(K) + (self.nu[0]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*(np.exp(lam*x1)-1.) + ( (self.nu[0]/self.alpha[0][0])*np.log((1-r1)*K) + (self.nu[0]*max(0,(self.T - Tfin))*(1-r1)*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1) + ( (self.nu[0]/self.alpha[0][0])*np.log(r3*K) + (self.nu[0]*max(0,(self.T - Tfin))*r3*K) )*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1) + ( (self.nu[0]/self.alpha[0][0])*np.log((1-r1)*r4*K) + (self.nu[0]*max(0,(self.T - Tfin))*(1-r1)*r4*K) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1)*(np.exp(lam*z1)-1))
            y0starmutfav -= ( (self.nu[1]/self.alpha[0][0])*np.log(self.ratiolen*K) + (self.nu[1]*max(0,(self.T - Tfin))*self.ratiolen*K) )*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*x1)-1.)*(np.exp(lam*z0)-1.)
            y0starmutunfav = self.T*self.nu[0]*lam*x1*np.exp(-lam)*np.exp(lam*(x0+x1+z1))*(np.exp(lam*z0)-1.)
            y0starmutunfav -= self.T*self.nu[1]*lam*z0*np.exp(-lam)*np.exp(lam*(x0+z0+z1))
        if z0 == 0. or self.nu[0]==0.: #Case when there is no mutations of replicases
            y0starmutfav = 0.
            y0starmutunfav = 0.
            x1 = 0.
            z1 = 0.
            value = max((y0starnomut+y0starmutfav+y0starmutunfav)/nstar,0.)
        self.y0 = y0starnomut+y0starmutfav+y0starmutunfav
        value = max((y0starnomut+y0starmutfav+y0starmutunfav)/nstar, 0.)
        ##print('y0 ', self.y0)
        return value
    
    def z1star(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        r1 = self.ratio1
        r2 = self.ratio2
        r3 = self.ratio3
        r4 = self.ratio4
        #nstar = lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #growth
        if self.tradeoff == 0:
            nstar = lam + K*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) + K*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.)
            y1starnomut = lam * z1 + (K/2)*self.ratiolen*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.)
        elif self.tradeoff == 1:
            nstar = lam + K*np.exp(-lam)*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z1)-1.) + K*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + (K/2)*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2) * np.exp(-lam) * (np.exp(lam*z1) - 1) * (np.exp(lam*x1)*(np.exp(lam*x0)-1.) + np.exp(lam*z0)*(np.exp(lam*x1)-1.))
            y1starnomut = lam * z1 + (K/2)*self.ratiolen * np.exp(-lam) * (np.exp(lam*z1) - 1) * (np.exp(lam*x1)*(np.exp(lam*x0)-1.) + np.exp(lam*z0)*(np.exp(lam*x1)-1.))
        elif self.tradeoff == 2:
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * (1. + 1/2 * ( (np.exp(lam* x1) - 1.) + (np.exp(lam* z1) - 1.) + 2/3 * (np.exp(lam* x1) - 1.) * (np.exp(lam* z1) - 1.) ) ) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( np.exp(lam*z0) * (1 + 1/2 *(np.exp(lam*z1) - 1.)) + 1/2 * (np.exp(lam*x0) - 1.) + 1/3 * (np.exp(lam*x0) - 1.) * (np.exp(lam*z1) - 1.)) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1 + z1)) * (np.exp(lam * x0) - 1.) * (np.exp(lam * z0) - 1.) + K*self.ratiolen/2 * np.exp(-lam) * ((np.exp(lam * x0) - 1.) * (np.exp(lam * z1) - 1.) + np.exp(lam * z0)*(np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.) + 2/3 * (np.exp(lam * x0) - 1.) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.))
            y1starnomut = K/2 * np.exp(-lam)*self.ratiolen * ((np.exp(lam * x0) - 1.) * (np.exp(lam * z1) - 1.) + np.exp(lam * z0)*(np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.) + 2/3 * (np.exp(lam * x0) - 1.) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.)) + lam * z1
        else:
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * ( np.exp(lam * z0) + r1 * (np.exp(lam * x1) - 1.) + r2 * (np.exp(lam * z1) - 1) + r1 * r4 * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( 1 + (1 - r1) * (np.exp(lam * x0) - 1) + r3 * (np.exp(lam * z1) - 1) + (1- r1) * r4 * (np.exp(lam * x0) - 1) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x0 + z1)) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z0) - 1.) + K * np.exp(-lam)*self.ratiolen * (np.exp(lam * z1) - 1) * ((1 - r2) * (np.exp(lam * x0) - 1) + (1 - r3) * (np.exp(lam * x1) - 1) + (1 - r4) * (np.exp(lam * x0) - 1) * (np.exp(lam * x1) - 1) ) 
            y1starnomut = K * np.exp(-lam)*self.ratiolen * (np.exp(lam * z1) - 1) * ((1 - r2) * (np.exp(lam * x0) - 1) + (1 - r3) * (np.exp(lam * x1) - 1) + (1 - r4) * (np.exp(lam * x0) - 1) * (np.exp(lam * x1) - 1) ) + lam * z1
        #mutations
        T = self.T
        Tfin = min(self.T, K/((K+1)*self.alpha[0][0]))
        if self.tradeoff ==0:
            y1starmutfav = (self.nu[1]*np.log(1/(1-self.alpha[0][0]*Tfin))/self.alpha[1,1] + self.nu[1]*K*max(0,(self.T - Tfin)))*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0)+np.exp(lam*z0)-1.) + lam*x1*self.nu[1]*max(0,(self.T - Tfin))*np.exp(-lam)*np.exp(lam*x1)*(np.exp(lam*x0)+np.exp(lam*z0)-1.)
            y1starmutunfav = lam*x1*self.nu[1]*self.T*np.exp(-lam)*np.exp(lam*x1)*((np.exp(lam*z1)-1.)*np.exp(lam*(x0+z0)) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.))
        elif self.tradeoff ==1:
            y1starmutfav = (self.nu[1]*np.log(1/(1-self.alpha[0][0]*Tfin))/self.alpha[1,1] + self.nu[1]*K*max(0,(self.T - Tfin)))*np.exp(-lam)*(np.exp(lam*x1)-1.)*(np.exp(lam*x0)+np.exp(lam*z0)-1.) + lam*x1*self.nu[1]*max(0,(self.T - Tfin))*np.exp(-lam)*np.exp(lam*x1)*(np.exp(lam*x0)+np.exp(lam*z0)-1.)
            y1starmutunfav = lam*x1*self.nu[1]*self.T*np.exp(-lam)*np.exp(lam*x1)*((np.exp(lam*z1)-1.)*np.exp(lam*(x0+z0)) + (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.))
            ##print(y1starmutfav, y1starmutunfav)
        elif self.tradeoff == 2:
            y1starmutfav = ( ( (self.nu[1]/self.alpha[1][1])*np.log(K) + (self.nu[1]*max(0,(self.T - Tfin))*K) )*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1)-1.) + ( (self.nu[1]/self.alpha[1][1])*np.log(K/2) + (self.nu[1]*max(0,(self.T - Tfin))*K/2) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.) + ( (self.nu[1]/self.alpha[1][1])*np.log(K/3) + (self.nu[1]*max(0,(self.T - Tfin))*K/3) )*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*x1)-1.)*(np.exp(lam*z1)-1.)  + (self.nu[1]*max(0,(self.T - Tfin))*lam*x1)*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)+np.exp(lam*z0)-1) )
            y1starmutunfav = self.T*self.nu[1]*lam*x1*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.)
            ##print(y1starmutfav,y1starmutunfav)
        else:
            y1starmutfav = ( (self.nu[1]/self.alpha[0][0])*np.log(self.ratiolen*K) + (self.nu[1]*max(0,(self.T - Tfin))*self.ratiolen*K) )*np.exp(-lam)*np.exp(lam*(x0+z1))*(np.exp(lam*x1)-1.)*(np.exp(lam*z0)-1.)
            y1starmutunfav = self.T*self.nu[1]*lam*z0*np.exp(-lam)*np.exp(lam*(x0+z0+z1))
        if z0 == 0. or self.nu[0]==0.: #Case when there is no mutations of replicases
            y1starmutfav = 0.
            y1starmutunfav = 0.
        else:
            self.ymutcount[0] += y1starmutfav
            self.ymutcount[1] += y1starmutunfav
        self.y1 = y1starnomut + y1starmutfav + y1starmutunfav
        value = max((y1starnomut + y1starmutfav + y1starmutunfav)/nstar, 0.)
        ##print('y1 ', self.y1)
        return value
    
    def lamdastar(self):
        x0 = self.x0
        x1 = self.x1
        z0 = self.z0
        z1 = self.z1
        K = self.K
        d = self.d
        lam = self.lamda
        r1 = self.ratio1
        r2 = self.ratio2
        r3 = self.ratio3
        r4 = self.ratio4
        if self.tradeoff == 0:
            nstar = lam + K*np.exp(-lam)*np.exp(lam*(x1+z1))*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.) + K*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2)*np.exp(-lam)*np.exp(lam*z0)*(np.exp(lam*x1) - 1.)*(np.exp(lam*z1) - 1.)
            self.n = nstar
        elif self.tradeoff == 1:
            nstar = lam + K*np.exp(-lam)*(np.exp(lam*x0)-1.) + (K/2)*np.exp(-lam)*(np.exp(lam*x0)-1.)*(np.exp(lam*z1)-1.) + K*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + (K/2)*np.exp(-lam)*(np.exp(lam*x1) - 1)*(np.exp(lam*z1) - 1)*(np.exp(lam*x0) + np.exp(lam*z0) - 1) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1+z1)) * (np.exp(lam*x0)-1.)*(np.exp(lam*z0)-1.) + (K*self.ratiolen/2) * np.exp(-lam) * (np.exp(lam*z1) - 1) * (np.exp(lam*x1)*(np.exp(lam*x0)-1.) + np.exp(lam*z0)*(np.exp(lam*x1)-1.))
            self.n = nstar
        elif self.tradeoff == 2:
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * (1. + 1/2 * ( (np.exp(lam* x1) - 1.) + (np.exp(lam* z1) - 1.) + 2/3 * (np.exp(lam* x1) - 1.) * (np.exp(lam* z1) - 1.) ) ) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( np.exp(lam*z0) * (1 + 1/2 *(np.exp(lam*z1) - 1.)) + 1/2 * (np.exp(lam*x0) - 1.) + 1/3 * (np.exp(lam*x0) - 1.) * (np.exp(lam*z1) - 1.)) + K*self.ratiolen * np.exp(-lam) * np.exp(lam*(x1 + z1)) * (np.exp(lam * x0) - 1.) * (np.exp(lam * z0) - 1.) + K*self.ratiolen/2 * np.exp(-lam) * ((np.exp(lam * x0) - 1.) * (np.exp(lam * z1) - 1.) + np.exp(lam * z0)*(np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.) + 2/3 * (np.exp(lam * x0) - 1.) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1.))
            self.n = nstar
        else:
            nstar = lam + K * np.exp(-lam) * (np.exp(lam * x0) - 1.) * ( np.exp(lam * z0) + r1 * (np.exp(lam * x1) - 1.) + r2 * (np.exp(lam * z1) - 1) + r1 * r4 * (np.exp(lam * x1) - 1.) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam) * (np.exp(lam * x1) - 1.) * ( 1 + (1 - r1) * (np.exp(lam * x0) - 1) + r3 * (np.exp(lam * z1) - 1) + (1- r1) * r4 * (np.exp(lam * x0) - 1) * (np.exp(lam * z1) - 1)) + K * np.exp(-lam)*self.ratiolen * np.exp(lam*(x0 + z1)) * (np.exp(lam * x1) - 1.) * (np.exp(lam * z0) - 1.) + K * np.exp(-lam)*self.ratiolen * (np.exp(lam * z1) - 1) * ((1 - r2) * (np.exp(lam * x0) - 1) + (1 - r3) * (np.exp(lam * x1) - 1) + (1 - r4) * (np.exp(lam * x0) - 1) * (np.exp(lam * x1) - 1) )
            self.n = nstar
        #nstar = lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)
        #self.n = lam + K*np.exp(-lam)*np.exp(lam*(z0+z1))*(np.exp(lam*(x0+x1))-1.)

        ##print('n/d ', nstar/d)
        return nstar/d
    
    def update(self):
        x0, x1, z0, z1, lamda = self.x0star(), self.x1star(), self.z0star(), self.z1star(), self.lamdastar()
        self.x0 = x0
        if self.mu[0]>0.:
            self.x1 = x1
        else:
            self.x1 = 0.
        self.z0 = z0
        ##print(z0)
        if self.nu[0]>0.:
            self.z1 = z1
        else:
            self.z1 = 0.
        self.lamda = lamda
        self.mmutant = np.array([x0, x1])
        self.ymutant = np.array([z0, z1])
    
    def evol(self):
        hist_evol = []
        hist_evol_numb = []
        for k in range(self.repet):
            self.mmutcount = [0,0]
            self.ymutcount = [0,0]
            ##print(self.m0+self.m1, self.m0+self.m1+self.y0+self.y1, self.n)
            hist_evol_numb.append([[self.m0, self.m1],[self.y0, self.y1]])
            ##print(hist_evol_numb[-1])
            ##print(r'sum = ', self.x0+self.x1+self.z0+self.z1)
            ##print(r'x_0=',self.x0,r'x_1=',self.x1,r'z_0=',self.z0,r'z_1=',self.z1)
            hist_evol.append([self.mmutant,self.ymutant])
            mcop, ycop = self.mmutant.copy(), self.ymutant.copy()
            self.update()
            if self.x0+self.x1+self.z0+self.z1>1.1:
                ##print(self.d, self.K)
                hist_evol.append([mcop,ycop])
        return np.array(hist_evol), np.array(hist_evol_numb)