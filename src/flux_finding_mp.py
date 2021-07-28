#testing editing
#can you see this yahui?
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar,minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from scipy import stats
import random as rd
seed=1000
rd.seed(seed)
np.random.seed(seed)


def labelingModel_addG3P(p,t,f,c,I,e):
    ratio = p[-1]/c[-1]
    dpdt = [e[0] + f[0]*ratio - (e[0]+f[0])*p[0]/c[0],
            e[2] + f[1]*ratio - (e[2] + f[1])*p[1]/c[1],
            e[1] + f[2]*ratio - (e[1]+f[2])*p[2]/c[2],
            -1*(f[0]+f[1]+f[2])*(ratio - I(t))]
    return dpdt

def labelingModel(p,t,f,c,I,e):
    ratio = p[-1]/c[-1]
    dpdt = [e[0] + f[0]*ratio - (e[0]+f[0])*p[0]/c[0],
            f[1]*ratio - (f[1])*p[1]/c[1],
            e[1] + f[2]*ratio - (e[1]+f[2])*p[2]/c[2],
            -1*(f[0]+f[1]+f[2])*(ratio - I(t))]
    return dpdt

def lacconstraint(x,maxFlux):
    return maxFlux - x[2] - x[4]

def gluconstraint(x,maxFlux):
    return 2*maxFlux - np.sum(x[:3])
    
def integrateLabelingModel_addG3P(t,flux,conc,a,e,init = 1):
    inter = lambda z: a[0]*np.exp(a[1]*z) + a[2]
    return np.concatenate(([[inter(tt)] for tt in t],np.array([np.divide(x,conc) for x in odeint(labelingModel_addG3P,np.array(conc),t,args=(flux,conc,inter,e))])),1)

def integrateLabelingModel(t,flux,conc,a,e,init = 1):
    inter = lambda z: a[0]*np.exp(a[1]*z) + a[2]
    return np.concatenate(([[inter(tt)] for tt in t],np.array([np.divide(x,conc) for x in odeint(labelingModel,np.array(conc),t,args=(flux,conc,inter,e))])),1)
    
def sse(p,e):
  return np.square(np.subtract(p,e)).mean()

def stochSSE(p,e):
  y = 0
  return np.sum(np.power(np.subtract(p[:,y],e[:,y]),2))


def findFlux(p,t,conc,lacE,gluUptake,initialParams = np.random.rand(4,1),q = False,convergence = 0.003):

    e1 = lacE*(p[-1,3]-p[-1,0])/(1-p[-1,0])
    f2 = lacE - e1
    
    a = fitSource(t,p[:,0])
    good = False
    bounds = [(None,None) for _ in range(9)]
    bounds[0] = (0,None)
    bounds[1] = (0,None)
    bounds[2] = (0,None)
    bounds[3] = (0,None)
    bounds[4] = (0,None)
    bounds[5] = (0,None)
    bounds = tuple(bounds)
    params = np.array(initialParams)
    params = [params[0],params[1],f2,params[2],e1,params[3]]
    if True:  
       fitted = minimize(lambda x: sse(p,integrateLabelingModel(t,x[:3],
                                                                np.append(conc[:-1],x[5:6]),x[6:],x[3:5])[:,:-1]),np.append(params,a),
                         method = 'SLSQP',constraints = [{'type': 'eq', 'fun': lambda x :lacconstraint(x,lacE)},
                                                         {'type': 'ineq', 'fun': lambda x :gluconstraint(x,gluUptake)}])#,

       good = fitted.success
      
    if good:
        params = fitted.x
        if q:
            q.put(np.append(params,fitted.fun))

        return np.append(params,fitted.fun)
    else:
        return -1
  
def findFlux_addG3P(p,t,conc,lacE,gluUptake,initialParams = np.random.rand(5,1),q = False,convergence = 0.003):

    e1 = lacE*(p[-1,3]-p[-1,0])/(1-p[-1,0])
    f2 = lacE - e1
    
    a = fitSource(t,p[:,0])
    good = False

    params = np.array(initialParams)
    params = [params[0],params[1],f2,params[2],e1,params[3],params[4]]
    if True:
       
       fitted = minimize(lambda x: sse(p,integrateLabelingModel_addG3P(t,x[:3],
                                                                np.append(conc[:-1],x[6:7]),x[7:],x[3:6])[:,:-1]),np.append(params,a),
                         method = 'SLSQP',constraints = [{'type': 'eq', 'fun': lambda x :lacconstraint(x,lacE)},
                                                         {'type': 'ineq', 'fun': lambda x :gluconstraint(x,gluUptake)}])
       good = fitted.success

       if not good: 
          print("failed")

    if good:
        params = fitted.x
        if q:
            q.put(np.append(params,fitted.fun))

        return np.append(params,fitted.fun)
    else:
        return -1


	
def removeBadSol(fluxes,cutoff=0.005,ci=95,target=100):
  
    fluxes = [x for x in fluxes if x[-1] < cutoff]
    if len(fluxes) > target:
      fluxes = rd.sample(fluxes,target)
    fluxes = np.array(fluxes)
    intervalParams = []
    interval = []
    for x in range(3):
      temp = list(fluxes[:,x])
      maxi = np.percentile(temp,100-((100-ci)/2),interpolation="nearest")
      mini = np.percentile(temp,(100-ci)/2,interpolation="nearest")
      indOfMax = temp.index(maxi)
      indOfMin = temp.index(mini)
      intervalParams.append([fluxes[indOfMin,:],fluxes[indOfMax,:]])
      interval.append([mini,maxi]) 
      

    return interval,intervalParams,fluxes  

def fitSource(t,l):
    good = False
    errorFunc = lambda x: sse([x[0]*np.exp(x[1]*tt)+x[2] for tt in t],l,)
    while not good:
      sol = minimize(errorFunc,[0,0,0])
      good = sol.success
    x = sol.x
    return x

# def abortable_worker(args):
#     q = Queue()
#     print(args)
#     t=Thread(findFlux
#     p = ThreadPool(1)
#     res = p.apply_async(func,[list(args)])
#     try:
#         out = res.get(timeout)  # Wait timeout seconds for func to complete.
#         return out
#     except multiprocessing.TimeoutError:
#         print("Aborting due to timeout")
#         return -1
