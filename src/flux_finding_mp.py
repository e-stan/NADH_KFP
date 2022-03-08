from scipy.integrate import odeint
from scipy.optimize import minimize
import numpy as np
import random as rd
#seed=1000
#rd.seed(seed)
#np.random.seed(seed)
from multiprocessing import Manager,Pool
from threading import Thread
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d

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

def labelingModel(state,t,fluxes,concentrations,dhap_unlabeled,unlabeled_source_fluxes):
    #[ldh, g3pshuttle,mdh]
    nadh_frac = state[3]/concentrations["NADH"]
    dpdt = [unlabeled_source_fluxes[0] + fluxes[0]*nadh_frac - (fluxes[0]+unlabeled_source_fluxes[0])*state[0]/concentrations["Lactate"],
            unlabeled_source_fluxes[1] + fluxes[1]*nadh_frac*dhap_unlabeled(t) - (fluxes[1]+unlabeled_source_fluxes[1])*state[1]/concentrations["G3P"],
            unlabeled_source_fluxes[2] + fluxes[2]*nadh_frac - (fluxes[2]+unlabeled_source_fluxes[2]) * state[2]/concentrations["Malate"],
            unlabeled_source_fluxes[3] + fluxes[3]*dhap_unlabeled(t) - (fluxes[3]+unlabeled_source_fluxes[3])*nadh_frac
            ]
    return dpdt

def lacconstraint(fluxes,maxFlux,labeled_contribution):
    return maxFlux - fluxes[0] / labeled_contribution

def gluconstraint(fluxes,maxFlux,labeled_contribution):
    return 2*maxFlux - fluxes[3]/labeled_contribution

def nadhBalance(fluxes,labeled_contribution):
    return fluxes[3] / labeled_contribution - np.sum(fluxes[:3])

def g3pModel(nadh,gap,c):
    return [(1-c) + (c)*(1-nadh)*(1-gap),
            (c) * ((1-nadh) * gap + nadh * (1-gap)),
            (c) * (nadh) * (gap)]

def g3plabelingModel(state,t,g3p_flux,unlabeled_flux,conc,nadh,gap):

    dpdt = [unlabeled_flux + g3p_flux*gap(t)*nadh(t) - (unlabeled_flux+g3p_flux)*state[0]/conc,
            g3p_flux*((1-gap(t))*nadh(t) + gap(t) * (1-nadh(t))) - (unlabeled_flux+g3p_flux)*state[1]/conc,
            g3p_flux*(1-gap(t))*(1-nadh(t)) - (unlabeled_flux+g3p_flux)*state[2]/conc
            ]
    return dpdt

def integrateG3PLabelingModel(t,g3p_flux,labeled_fraction,conc,nadh,dhap,initial_state):
    totalFlux = g3p_flux/labeled_fraction
    unlabeled_flux = totalFlux - g3p_flux
    result = odeint(g3plabelingModel,initial_state,t,args=(g3p_flux,unlabeled_flux,conc,nadh,dhap),tfirst=False)
    result = result / conc
    return result

def calculateCorrectionFactorForNADH(gap,nadh,g3p):
    sol = minimize(lambda x:sse(g3p,g3pModel(x[0],gap,x[1])),x0=[nadh/2,0],bounds=[(nadh,1),(0,1)])
    c = nadh / sol.x[0]
    c = 1/c
    return c,sol.x[1]


def integrateLabelingModel(t,fluxes,concs,dhap_params,labeled_fractions,initial_state):
    dhap_unlabeled = lambda z: exponetialCurve(z,dhap_params)
    unlabeled_source_fluxes = [(1-c)/c * f for c,f in zip(labeled_fractions,fluxes)]
    result = odeint(labelingModel,initial_state,t,args=(fluxes,concs,dhap_unlabeled,unlabeled_source_fluxes),tfirst=False)
    for x,label in zip(range(result.shape[1]),["Lactate","G3P","Malate","NADH"]):
        result[:,x] = result[:,x] / concs[label]
    return result

def sse(p,e):
  return np.square(np.subtract(p,e)).mean()

def findFlux(data, t, conc, lacE, gluUptake, initialFluxes = np.random.random(4), q = False):
    lastT = np.max(t)
    lastT = [x for x in range(len(t)) if abs(lastT-t[x]) < 1e-5]
    filt = data.loc[lastT,:]

    #define unlabeled fractions
    labeled_contributions = np.zeros(4)

    #correct nadh labeling from g3p
    corr_factor, labeled_contributions[1] = calculateCorrectionFactorForNADH(filt["L_gap"].values.mean(), filt["L_nadh"].values.mean(), [filt["UL_g3p"].values.mean(),filt["L_g3p_M+1"].values.mean(),filt["L_g3p_M+2"].values.mean()])
    filt["L_nadh"] = corr_factor * filt["L_nadh"]
    data["L_nadh"] = corr_factor * data["L_nadh"]
    filt["UL_nadh"] = 1 - filt["L_nadh"]
    data["UL_nadh"] = 1 - data["L_nadh"]

    #lactate
    labeled_contributions[0] = (filt["L_lac"].values.mean()/filt["L_nadh"].values.mean())

    #malate
    labeled_contributions[2] = (filt["L_malate"].values.mean()/filt["L_nadh"].values.mean())

    #nadh
    labeled_contributions[3] = (filt["L_nadh"].values.mean()/filt["L_gap"].values.mean())

    #define initial conditions
    firstT = np.min(t)
    firstT = [x for x in range(len(t)) if abs(firstT-t[x]) < 1e-5]
    initialState = [np.mean(data.loc[firstT,label])*c for label,c in zip(["UL_lac","UL_g3p","UL_malate","UL_nadh"],[conc["Lactate"],conc["G3P"],conc["Malate"],conc["NADH"]])]

    #fit curve to dhap (gap) data
    dhap_params = fitSource(t, data["UL_gap"])

    #initalize bounds on variables
    bounds = tuple([(0,None) for _ in range(len(initialFluxes))])

    #data to fit
    dataForComparision = data[["UL_lac","UL_g3p","UL_malate","UL_nadh"]].to_numpy()

    #correct for unsatisfiable constraint
    if np.sum(initialFluxes[:3]) > gluUptake:
        initialFluxes[:3] = initialFluxes[:3]/gluUptake

    #define initial flux
    initialFluxes[3] = np.sum(initialFluxes[:3]) * labeled_contributions[3]

    #perform fit
    fitted = minimize(lambda x: sse(dataForComparision, integrateLabelingModel(t, x,conc,dhap_params,labeled_contributions,initialState)),x0=initialFluxes,
                    constraints = [{'type': 'ineq', 'fun': lambda x :gluconstraint(x,gluUptake,labeled_contributions[3])},
                                   {'type': 'eq', 'fun': lambda x :nadhBalance(x,labeled_contributions[3])},
                                   {'type': 'eq', 'fun': lambda x :lacconstraint(x,lacE,labeled_contributions[0])}],bounds=bounds)


    #return result
    good = fitted.success

    if good:
        params = fitted.x
        if q:
            q.put(1)

        return params,labeled_contributions,fitted.fun,data
    else:
        if q:
            q.put(1)
        return -1



def generateSyntheticData(ts):
    fluxes = np.random.random(4)
    labeled_contributions = np.random.random(4)
    fracOfGlycolysis = np.random.random()
    glycolysis = 10 * np.random.random()
    gapdh = glycolysis * fracOfGlycolysis
    fluxes[3] = gapdh
    fluxes[:3] = fluxes[:3] * gapdh / labeled_contributions[3] / np.sum(fluxes[:3])

    dhap_params = np.random.random(3)
    dhap_params[1] = -1*dhap_params[1]
    dhap_params[[0,2]] = dhap_params[[0,2]] / np.sum(dhap_params[[0,2]])

    conc = np.random.random(4)
    initialState = conc

    conc = {label:con for label,con in zip(["Lactate","G3P","Malate","NADH"],conc)}

    dynamics = integrateLabelingModel(ts, fluxes, conc, dhap_params, labeled_contributions, initialState)

    df = pd.DataFrame(data=dynamics,columns=["UL_lac","UL_g3p","UL_malate","UL_nadh"])
    df["UL_gap"] = exponetialCurve(ts,dhap_params)

    df["L_lac"] = 1-df["UL_lac"]
    df["L_malate"] = 1-df["UL_malate"]
    df["L_nadh"] = 1-df["UL_nadh"]
    df["L_gap"] = 1-df["UL_gap"]

    nadh = interp1d(ts,df["UL_nadh"].values,bounds_error=False,fill_value="extrapolate")

    initialState = np.array([initialState[1],0,0])


    result = integrateG3PLabelingModel(ts, fluxes[1], labeled_contributions[1], conc["G3P"], nadh, lambda x: exponetialCurve(x,dhap_params), initialState)

    df["L_g3p_M+1"] = result[:,1]
    df["L_g3p_M+2"] = result[:,2]

    lacE = fluxes[0] / labeled_contributions[0]

    return df,lacE,glycolysis,fluxes,conc,labeled_contributions


def simulateDataAndInferFlux(ts,numIts,q=None):
    data,lacE,glycolysis,fluxes,conc,labeled_contributions = generateSyntheticData(ts)
    args = [[data,ts,conc,lacE,glycolysis,np.random.random(4)] for _ in range(numIts)]
    startingParams = startConcurrentTask(findFlux,args,1,"finding best fit",len(args),verbose=False)
    startingParams = [x[:-1] for x in startingParams if type(x) != type(-1)]
    if len(startingParams) > 0:
        startingParams.sort(key=lambda x: x[-1])
        bestParams = np.append(startingParams[0][0],startingParams[0][-1])
    else:
        bestParams = -1
    if type(q) != type(None):
        q.put(1)
    return fluxes,bestParams
	
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
    #while not good:
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



