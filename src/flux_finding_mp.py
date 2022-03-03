from scipy.integrate import odeint
from scipy.optimize import minimize
import numpy as np
import random as rd
seed=1000
rd.seed(seed)
np.random.seed(seed)
from multiprocessing import Manager,Pool
from threading import Thread
import pandas as pd
import scipy.stats as stats

def startConcurrentTask(task,args,numCores,message,total,chunksize="none",verbose=True):
    if verbose:
        m = Manager()
        q = m.Queue()
        args = [a + [q] for a in args]
        t = Thread(target=updateProgress, args=(q, total, message))
        t.start()
    if numCores > 1:
        p = Pool(numCores)
        if chunksize == "none":
            res = p.starmap(task, args)
        else:
            res = p.starmap(task, args, chunksize=chunksize)
        p.close()
        p.join()
    else:
        res = [task(*a) for a in args]
    if verbose: t.join()
    return res

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def updateProgress(q, total,message = ""):
    counter = 0
    while counter != total:
        if not q.empty():
            q.get()
            counter += 1
            printProgressBar(counter, total,prefix = message, printEnd="")

def labelingModel(state,t,fluxes,concentrations,dhap_unlabeled,nadh_unlabeled):
    #[ldh, g3pshuttle,mdh]
    dpdt = [fluxes[6] + fluxes[5]*nadh_unlabeled(t) - (fluxes[6]+fluxes[5])*state[0]/concentrations["Lactate"],
            fluxes[4] + fluxes[3]*nadh_unlabeled(t)*dhap_unlabeled(t) - (fluxes[4]+fluxes[3])*state[1]/concentrations["G3P"],
            fluxes[1] + fluxes[2]*nadh_unlabeled(t) - (fluxes[2] + fluxes[1]) * state[2]/concentrations["Malate"]]
    return dpdt

def lacconstraint(fluxes,maxFlux):
    return maxFlux - fluxes[6] - fluxes[5]

def gluconstraint(fluxes,maxFlux):
    return 2*maxFlux - (fluxes[5] + fluxes[3] + fluxes[2])

def integrateLabelingModel(t,fluxes,concs,dhap_params,nadh_params,initial_state):
    nadh_unlabeled = lambda z: exponetialCurve(z,nadh_params)#nadh_params[0]*np.exp(nadh_params[1]*z) + nadh_params[2]
    dhap_unlabeled = lambda z: exponetialCurve(z,dhap_params)#dhap_params[0]*np.exp(dhap_params[1]*z) + dhap_params[2]
    result = odeint(labelingModel,initial_state,t,args=(fluxes,concs,dhap_unlabeled,nadh_unlabeled),tfirst=False)
    for x,label in zip(range(result.shape[1]),["Lactate","G3P","Malate"]):
        result[:,x] = result[:,x] / concs[label]
    return result

def sse(p,e):
  return np.square(np.subtract(p,e)).mean()

def findFlux(data, t, conc, lacE, gluUptake, initialFluxes = np.random.random(7), q = False):
    lastT = np.max(t)
    lastT = [x for x in range(len(t)) if abs(lastT-t[x]) < 1e-5]

    initialFluxes[6] = lacE * (np.mean(data.loc[lastT, "UL_lac"].values) - np.mean(data.loc[lastT, "UL_nadh"].values)) / (1 - np.mean(data.loc[lastT, "UL_nadh"].values))
    initialFluxes[5] = lacE - initialFluxes[6]

    firstT = np.min(t)
    firstT = [x for x in range(len(t)) if abs(firstT-t[x]) < 1e-5]
    initialState = [np.mean(data.loc[firstT,label])*c for label,c in zip(["UL_lac","UL_g3p","UL_malate"],[conc["Lactate"],conc["G3P"],conc["Malate"]])]
    
    dhap_params = fitSource(t, data["UL_gap"])
    nadh_params = fitSource(t, data["UL_nadh"])

    bounds = tuple([(0,None) for _ in range(len(initialFluxes))])

    dataForComparision = data[["UL_lac","UL_g3p","UL_malate"]].to_numpy()

    fitted = minimize(lambda x: sse(dataForComparision, integrateLabelingModel(t, x,conc,dhap_params,nadh_params,initialState)),x0=initialFluxes,
                     method = 'SLSQP',constraints = [{'type': 'eq', 'fun': lambda x :lacconstraint(x,lacE)},
                                                     {'type': 'ineq', 'fun': lambda x :gluconstraint(x,gluUptake)}],bounds=bounds)#,

    good = fitted.success

    if good:
        params = fitted.x
        params[0] = params[1] + params[4] + params[6]
        if q:
            q.put(1)

        return np.append(params,fitted.fun)
    else:
        if q:
            q.put(1)
        return -1

	
def removeBadSol(fluxes,cutoff=0.005,ci=95,target=100):
  
    fluxes = [x for x in fluxes if x[-1] < cutoff]
    if len(fluxes) > target:
      fluxes = rd.sample(fluxes,target)
    fluxes = np.array(fluxes)
    intervalParams = []
    interval = []
    for x in range(fluxes.shape[1]-1):
      temp = list(fluxes[:,x])
      maxi = np.percentile(temp,100-((100-ci)/2),interpolation="nearest")
      mini = np.percentile(temp,(100-ci)/2,interpolation="nearest")
      indOfMax = temp.index(maxi)
      indOfMin = temp.index(mini)
      intervalParams.append([fluxes[indOfMin,:],fluxes[indOfMax,:]])
      interval.append([mini,maxi]) 
      

    return interval,intervalParams,fluxes

def exponetialCurve(t,params):
    return params[0] * np.exp(params[1] * t) + params[2]


def fitSource(t,l):
    good = False
    errorFunc = lambda x: sse(exponetialCurve(t,x),l)
    while not good:
      sol = minimize(errorFunc,[0,0,0])
      good = sol.success
    x = sol.x
    return x


def clip(val, max=1.0, min=0.0):
    if val > max:
        return max
    if val < min:
        return min
    return val

def resampledata(data,ts,q=False):
    cols = data.columns.values
    newDf = pd.DataFrame(data)
    data["t"] = ts
    for t in list(set(ts)):
        filt = data[data["t"] == t]
        for col in cols:
            vals = filt[col].values
            params = stats.t.fit(vals)
            newDf.at[filt.index.values,col] = np.array([clip(x,1.0,0.0) for x in stats.t.rvs(*params,size=len(vals))])
    if q:
        q.put(1)
    return newDf


def runMonteCarlo(data, t, conc, exc, gluUp, initialParams=np.random.rand(7, 1), numIts=100,numCores=4):
    # define monte carlo datasets

    datasets = startConcurrentTask(resampledata,[[data,t] for _ in range(numIts)],numCores,"resampling datasets",numIts)
    args = [[dataset,t,conc,exc,gluUp,initialParams] for dataset in datasets]

    fluxes = startConcurrentTask(findFlux,args,numCores,"running monte carlo",numIts)

    fluxes = [x for x in fluxes if type(x) != type(-1)]
    print(len(fluxes),"successful iterations complete")

    fluxes = np.array(fluxes)

    return fluxes



