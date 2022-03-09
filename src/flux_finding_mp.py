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

# def labelingModel(state,t,fluxes,concentrations,dhap_unlabeled,unlabeled_source_fluxes):
#     #[ldh, g3pshuttle,mdh]
#     nadh_frac = state[3]/concentrations["NADH"]
#     dpdt = [unlabeled_source_fluxes[0] + fluxes[0]*nadh_frac - (fluxes[0]+unlabeled_source_fluxes[0])*state[0]/concentrations["Lactate"],
#             unlabeled_source_fluxes[1] + fluxes[1]*nadh_frac*dhap_unlabeled(t) - (fluxes[1]+unlabeled_source_fluxes[1])*state[1]/concentrations["G3P"],
#             unlabeled_source_fluxes[2] + fluxes[2]*nadh_frac - (fluxes[2]+unlabeled_source_fluxes[2]) * state[2]/concentrations["Malate"],
#             unlabeled_source_fluxes[3] + fluxes[3]*dhap_unlabeled(t) - (fluxes[3]+unlabeled_source_fluxes[3])*nadh_frac
#             ]
#     return dpdt

def lactateEquation(state,t,flux,conc,unlabeled_flux,nadh,gap):
    return unlabeled_flux + flux * nadh(t) - (unlabeled_flux+flux) * state / conc

def malateEquation(state,t,flux,conc,unlabeled_flux,nadh,gap):
    return unlabeled_flux + flux * nadh(t) - (unlabeled_flux+flux) * state / conc

def g3pEquation(state,t,flux,conc,unlabeled_flux,nadh,gap):
    return unlabeled_flux + flux * nadh(t) * gap(t) - (unlabeled_flux+flux) * state / conc

def nadhEquation(state,t,flux,conc,unlabeled_flux,nadh,gap):
    return unlabeled_flux + flux * gap(t) - (unlabeled_flux+flux) * state / conc

def integrateModel(eq,t,args,init,conc):
    return odeint(eq, init, t, args=args, tfirst=False)/conc

# def lacconstraint(fluxes,maxFlux,labeled_contribution):
#     return maxFlux - fluxes[0] / labeled_contribution
#
# def gluconstraint(fluxes,maxFlux,labeled_contribution):
#     return 2*maxFlux - fluxes[3]/labeled_contribution

# def nadhBalance(fluxes,labeled_contribution):
#     return fluxes[3] / labeled_contribution - np.sum(fluxes[:3])

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
    sol = minimize(lambda x:sse(g3p,g3pModel(x[0],gap,x[1])),x0=[nadh/2,0],bounds=[(0,1),(0,1)])
    c = nadh / sol.x[0]
    c = 1/c
    return c,sol.x[1]


def integrateLabelingModel(t,fluxes,concs,dhap_params,labeled_fractions,initial_state):
    unlabeled_source_fluxes = [(1-c)/c * f for c,f in zip(labeled_fractions,fluxes)]
    dhap = lambda z: exponetialCurve(z,dhap_params)
    nadh = interp1d(np.linspace(min(t),max(t),100),
                    integrateModel(nadhEquation,np.linspace(min(t),max(t),100),(fluxes[3],concs['NADH'],unlabeled_source_fluxes[3],None,dhap),initial_state[3],concs["NADH"])[:,0],
                    bounds_error=False,fill_value="extrapolate")

    result = np.zeros((len(t),4))
    equations = [lactateEquation,g3pEquation,malateEquation,nadhEquation]
    labels = ["Lactate","G3P","Malate","NADH"]
    for x in range(4):
        result[:,x] = integrateModel(equations[x],t,(fluxes[x],concs[labels[x]],unlabeled_source_fluxes[x],nadh,dhap),np.array(initial_state[x]),concs[labels[x]])[:,0]
    return result

def sse(p,e):
  return np.square(np.subtract(p,e)).mean()

def findFlux(data, t, conc, lacE, gluUptake, initialFluxes = np.random.random(4), q = False):
    lastT = np.max(t)
    lastT = [x for x in range(len(t)) if abs(lastT-t[x]) < 1e-5]
    data = pd.DataFrame(data)
    filt = data.loc[lastT,:]

    fluxes = np.zeros(initialFluxes.shape)

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

    #define NADH concentration if not provided
    calculateNADHConc = False
    if "NADH" not in conc:
        conc["NADH"] = 1.0
        calculateNADHConc = True

    #define initial conditions
    firstT = np.min(t)
    firstT = [x for x in range(len(t)) if abs(firstT-t[x]) < 1e-5]
    initialState = [np.mean(data.loc[firstT,label])*c for label,c in zip(["UL_lac","UL_g3p","UL_malate","UL_nadh"],[conc["Lactate"],conc["G3P"],conc["Malate"],conc["NADH"]])]

    #fit curve to dhap (gap) data
    dhap_params = fitSource(t, data["UL_gap"])
    dhap = lambda x: exponetialCurve(x,dhap_params)

    #initalize bounds on variables
    #bounds = tuple([(0,None) for _ in range(len(initialFluxes))])

    errs = np.zeros(4)

    #fit NADH curve
    fitted = minimize(lambda x: sse(data["UL_nadh"].values,integrateModel(nadhEquation,t,
                    (x[0],conc["NADH"],(1-labeled_contributions[3])/labeled_contributions[3] * x[0],None,dhap),
                    initialState[3],conc["NADH"])[:,0]),x0=np.array([initialFluxes[3]]),method="Nelder-Mead",
                      options={"fatol":1e-9})

    fluxes[3] = fitted.x[0]
    errs[3] = fitted.fun
    nadh = interp1d(np.linspace(min(t),max(t),100),
                    integrateModel(nadhEquation,np.linspace(min(t),max(t),100),(fluxes[3],conc['NADH'],(1-labeled_contributions[3])/labeled_contributions[3] * fluxes[3],None,dhap),initialState[3],conc["NADH"])[:,0],
                    bounds_error=False,fill_value="extrapolate")    #

    #fit lactate, g3p, and malate
    equations = [lactateEquation,g3pEquation,malateEquation]
    labels1 = ["UL_lac","UL_g3p","UL_malate"]
    labels2 = ["Lactate","G3P","Malate"]
    for x in range(3):
        fitted = minimize(lambda z: sse(data[labels1[x]].values,integrateModel(equations[x],t,
                        (z[0],conc[labels2[x]],(1-labeled_contributions[x])/labeled_contributions[x] * z[0],nadh,dhap),
                        initialState[x],conc[labels2[x]])[:,0]),x0=np.array([initialFluxes[x]]),method="Nelder-Mead",
                          options={"fatol":1e-9})
        fluxes[x] = fitted.x[0]
        errs[x] = fitted.fun

    if calculateNADHConc:
        #fix flux
        fluxes[3] = np.sum(fluxes[:3]) * labeled_contributions[3]
        # fit NADH curve to calculate NADH pool size
        fitted = minimize(lambda x: sse(data["UL_nadh"].values, integrateModel(nadhEquation, t,
                                                                               (fluxes[3], x[0],
                                                                                (1 - labeled_contributions[3]) /
                                                                                labeled_contributions[3] * fluxes[3], None,
                                                                                dhap),
                                                                               initialState[3]*x[0], x[0])[:, 0]),
                          x0=np.array([np.random.random()]), method="Nelder-Mead",
                          options={"fatol": 1e-9})
        conc["NADH"] = fitted.x[0]


    integratedSolution = integrateLabelingModel(t,fluxes,conc,dhap_params,labeled_contributions,initialState)
    data["UL_lac"] = integratedSolution[:,0]
    data["L_lac"] = 1-integratedSolution[:,0]

    data["UL_g3p"] = integratedSolution[:,1]
    data["L_g3p"] = 1-integratedSolution[:,1]

    data["UL_malate"] = integratedSolution[:,2]
    data["L_malate"] = 1-integratedSolution[:,2]

    data["UL_nadh"] = integratedSolution[:,3]
    data["L_nadh"] = 1-integratedSolution[:,3]


    if q:
        q.put(1)

    return fluxes,labeled_contributions,data,conc,errs



def generateSyntheticData(ts,noise=.00):
    fluxes = np.random.random(4)
    labeled_contributions = .3 + .7 * np.random.random(4)
    #labeled_contributions[3] = 1.0
    fracOfGlycolysis = np.random.random()
    glycolysis = 10 * np.random.random()
    gapdh = glycolysis * fracOfGlycolysis
    fluxes[3] = gapdh
    fluxes[:3] = fluxes[:3] * gapdh / labeled_contributions[3] / np.sum(fluxes[:3])

    dhap_params = np.random.random(3)
    dhap_params[1] = -1*dhap_params[1] * 3
    dhap_params[[0,2]] = dhap_params[[0,2]] / np.sum(dhap_params[[0,2]])

    conc = np.random.random(4)
    initialState = conc

    switcher = [-1,1]

    conc = {label:con for label,con in zip(["Lactate","G3P","Malate","NADH"],conc)}

    dynamics = integrateLabelingModel(ts, fluxes, conc, dhap_params, labeled_contributions, initialState)

    #add noise to conc
    conc = {label:con+rd.choice(switcher) * con * noise * np.random.random() for label,con in conc.items()}

    #add noise to dyanamics
    for r in range(dynamics.shape[0]):
        for c in range(dynamics.shape[1]):
            dynamics[r,c] = dynamics[r,c] + rd.choice(switcher) * dynamics[r,c] * noise * np.random.random()


    df = pd.DataFrame(data=dynamics,columns=["UL_lac","UL_g3p","UL_malate","UL_nadh"])
    df["UL_gap"] = [v+v*rd.choice(switcher)*noise*np.random.random() for v in exponetialCurve(ts,dhap_params)]

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
    tmp = {key:val for key,val in conc.items() if "NADH" != key}
    args = [[data,ts,tmp,lacE,glycolysis,np.random.random(4)] for _ in range(numIts)]
    startingParams = startConcurrentTask(findFlux,args,1,"finding best fit",len(args),verbose=False)

    startingParams.sort(key=lambda x: np.sum(x[-1]))
    bestParams = startingParams[0]

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



