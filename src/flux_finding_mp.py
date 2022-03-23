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
from copy import deepcopy

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

def KIEConstantLactateMalateNADH(vhvd,x):
    """
    returns the KIE constant for LDH, MAD, and GAPDH reactions
    :param vhvd: reaction velocity of hydrogen / deuterium reaction
    :param x: substrate (NADH) labeleing percentage
    :return: KIE constant
    """
    return vhvd + x * (1-vhvd)

def KIEConstantG3P(vhvd_x,vhvd_y,vhvd_xy,x,y):
    """
    returns the KIE constant for G3PS reaction
    :param vhvd_x: reaction velocity of hydrogen / deuterium reaction when substrate 1 is labeled
    :param vhvd_y: reaction velocity of hydrogen / deuterium reaction when substrate 2 is labeled
    :param vhvd_xy: reaction velocity of hydrogen / deuterium reaction when both substrates are labeled
    :param x: substrate 1 labeling percentage
    :param y: substrate 2 labeling percentage
    :return: KIE constant
    """
    return (x*(1-y)/vhvd_x + (1-x)*y/vhvd_y)*vhvd_xy + x*y + ((1-x)*(1-y))*vhvd_xy

def CtConstantG3P(dhap,nadh,vhvd_dhap,vhvd_nadh,vhvd_nadh_dhap):
    """
    calculate ct constant (ft = ct * vh) for g3ps reaction
    :param dhap: dhap labeling %
    :param nadh: nadh labeling %
    :param vhvd_dhap: reaction velocity of hydrogen / deuterium reaction when dhap is labeled
    :param vhvd_nadh: reaction velocity of hydrogen / deuterium reaction when nadh is labeled
    :param vhvd_nadh_dhap: reaction velocity of hydrogen / deuterium reaction when both nadh and dhap are labeled
    :return: ct constant
    """
    return dhap * (1-nadh) / vhvd_dhap + (1-dhap) * nadh /vhvd_nadh + dhap * nadh / vhvd_nadh_dhap + (1-nadh) * (1-dhap)

def CtConstantMalateLactateNADH(x,vhvd):
    """
    calculate ct constant (ft = ct * vh) for malate, lactate and gapdh reactions
    :param x: labeling percentage of substrate (NADH)
    :param vhvd: reaction velocity of hydrogen / deuterium reaction when substrate is labeled
    :return: ct constant
    """
    return (1-x) + x/vhvd

def C0ConstantG3P(dhap,nadh,m2,vhvd_dhap,vhvd_nadh,vhvd_nadh_dhap):
    """
    calculate c0 constant (f0 = c0*vh) for G3P reaction
    :param dhap: dhap labeling % at ISS (isotopic steady state)
    :param nadh: nadh labeling # at ISS
    :param m2: M+2 labeleing percentage of G3P
    :param vhvd_dhap: reaction velocity of hydrogen / deuterium reaction when dhap is labeled
    :param vhvd_nadh: reaction velocity of hydrogen / deuterium reaction when nadh is labeled
    :param vhvd_nadh_dhap: reaction velocity of hydrogen / deuterium reaction when both nadh and dhap are labeled
    :return: c0 constant
    """

    return (dhap * nadh / vhvd_nadh_dhap + dhap * (1-nadh) / vhvd_dhap + (1-dhap) * nadh / vhvd_nadh + (1-dhap) * (1-nadh)) * ((dhap*nadh)/m2-KIEConstantG3P(vhvd_nadh,vhvd_dhap,vhvd_nadh_dhap,nadh,dhap))/KIEConstantG3P(vhvd_nadh,vhvd_dhap,vhvd_nadh_dhap,nadh,dhap)

def C0ConstantMalateLactateNADH(x,p,vhvd):
    """
    calculate c0 constant (f0 = c0 * vh) for MAS, LDH, and GAPDH reactions
    :param x: substrate (NADH) labeling % at ISS
    :param p: product labeling % at ISS
    :param vhvd: reaction velocity of hydrogen / deuterium reaction when substrate is labeled
    :return: c0 constant
    """
    return ((1-x)+x/vhvd)*((x/p)-KIEConstantLactateMalateNADH(vhvd,x))/KIEConstantLactateMalateNADH(vhvd,x)

def lactateEquation(state,t,vh,conc,unlabeled_flux,nadh,gap,vhvds):
    ct = CtConstantMalateLactateNADH((1-nadh(t)),vhvds["vhvd_nadh_ldh"])
    return unlabeled_flux + vh * nadh(t) - (unlabeled_flux+ct*vh) * state / conc

def malateEquation(state,t,vh,conc,unlabeled_flux,nadh,gap,vhvds):
    ct = CtConstantMalateLactateNADH((1-nadh(t)),vhvds["vhvd_nadh_mas"])
    return unlabeled_flux + vh * nadh(t) - (unlabeled_flux+ct*vh) * state / conc

def g3pEquation(state,t,vh,conc,unlabeled_flux,nadh,gap,vhvds):
    ct = CtConstantG3P((1-gap(t)),(1-nadh(t)),vhvds["vhvd_dhap_g3ps"],vhvds["vhvd_nadh_g3ps"],vhvds["vhvd_nadh_dhap_g3ps"])
    return unlabeled_flux + vh * nadh(t) * gap(t) - (unlabeled_flux+vh * ct) * state / conc

def nadhEquation(state,t,vh,conc,unlabeled_flux,nadh,gap,vhvds):
    ct = CtConstantMalateLactateNADH((1-gap(t)),vhvds["vhvd_gap_gapdh"])
    return unlabeled_flux + vh * gap(t) - (unlabeled_flux+ct * vh) * state / conc

def integrateModel(eq,t,args,init,conc):
    return odeint(eq, init, t, args=args, tfirst=False)/conc

# def lacconstraint(fluxes,maxFlux,labeled_contribution):
#     return maxFlux - fluxes[0] / labeled_contribution
#
# def gluconstraint(fluxes,maxFlux,labeled_contribution):
#     return 2*maxFlux - fluxes[3]/labeled_contribution

# def nadhBalance(fluxes,labeled_contribution):
#     return fluxes[3] / labeled_contribution - np.sum(fluxes[:3])

def g3pModel(nadh,gap,c,vhvds):
    ct = CtConstantG3P(gap,nadh,vhvds["vhvd_dhap_g3ps"],vhvds["vhvd_nadh_g3ps"],vhvds["vhvd_nadh_dhap_g3ps"])
    return np.array([c + (1-c)/ct*(1-gap)*(1-nadh),
            (1-c)/ct * (nadh * (1-gap)/vhvds["vhvd_nadh_g3ps"] + (1-nadh) * gap / vhvds["vhvd_dhap_g3ps"]),
            (1-c)/ct * nadh * gap / vhvds["vhvd_nadh_dhap_g3ps"]])

def g3plabelingModel(state,t,vh,unlabeled_flux,conc,nadh,gap,vhvds):
    ct = CtConstantG3P(1-gap(t),1-nadh(t),vhvds["vhvd_dhap_g3ps"],vhvds["vhvd_nadh_g3ps"],vhvds["vhvd_nadh_dhap_g3ps"])
    dpdt = [unlabeled_flux + vh*gap(t)*nadh(t) - (unlabeled_flux+ct*vh)*state[0]/conc,
            vh/vhvds["vhvd_nadh_g3ps"]*(1-nadh(t)) * gap(t) + vh/vhvds["vhvd_dhap_g3ps"]*nadh(t)*(1-gap(t)) - (unlabeled_flux+ct*vh)*state[1]/conc,
            vh/vhvds["vhvd_nadh_dhap_g3ps"]*(1-gap(t))*(1-nadh(t)) - (unlabeled_flux+ct*vh)*state[2]/conc
            ]
    return dpdt

def integrateG3PLabelingModel(t,g3p_flux,c0,conc,nadh,dhap,initial_state,vhvds):
    unlabeled_flux = c0 * g3p_flux
    result = odeint(g3plabelingModel,initial_state,t,args=(g3p_flux,unlabeled_flux,conc,nadh,dhap,vhvds),tfirst=False)
    result = result / conc
    return result

def calculateSteadyStateNADH(gap, g3p, vhvds, lb):
    if lb > gap:
        lb = 0

    sol = minimize(lambda x:sse(g3p,g3pModel(x[0],gap,x[1],vhvds)),x0=np.array([1,0]),bounds=[(lb,gap),(0,1)])
    nadh = sol.x[0]
    ct = CtConstantG3P(gap, nadh, vhvds["vhvd_dhap_g3ps"], vhvds["vhvd_nadh_g3ps"], vhvds["vhvd_nadh_dhap_g3ps"])
    c = sol.x[1]
    c0 = c * ct / (1-c)
    return nadh,c0

def labelingModelTotal(state,t,fluxes,unlabeled_fluxes,concs,dhap,vhvds):
    nadh = lambda z:state[3]/concs["NADH"]
    equations = [lactateEquation,g3pEquation,malateEquation,nadhEquation]
    dpdt = [equations[x](state[x],t,fluxes[x],concs[lab],unlabeled_fluxes[x],nadh,dhap,vhvds) for x,lab in zip(range(4),["Lactate","G3P","Malate","NADH"])]
    return dpdt

def integrateLabelingModel(t,fluxes,concs,dhap_params,c0s,vhvds,initial_state):

    unlabeled_source_fluxes = [c * f for c,f in zip(c0s,fluxes)]
    dhap = lambda z: exponetialCurve(z,dhap_params)
    sol = integrateModel(labelingModelTotal,t,(fluxes,unlabeled_source_fluxes,concs,dhap,vhvds),initial_state,1.0)
    result = np.zeros(sol.shape)
    labels = ["Lactate","G3P","Malate","NADH"]
    for x in range(4):
        result[:,x] = sol[:,x]/concs[labels[x]]
    return result

def sse(p,e,weights=None):
  if type(weights) == type(None):
      weights = np.ones(p.shape)
  return np.multiply(np.square(np.subtract(p,e)),weights).sum()

def findFlux(data, t, conc, lacE, gluUptake,vhvds, initialFluxes = np.random.random(4), q = False):
    #get data at last time point
    lastT = np.max(t)
    lastT = [x for x in range(len(t)) if abs(lastT-t[x]) < 1e-5]
    data = deepcopy(data)
    filt = data.loc[lastT,:]

    #get mapping of times to index
    uniqueTs = list(set(t))
    uniqueTs.sort()
    tMapper = {}
    for tt in uniqueTs:
        tMapper[tt] = []
        for x in range(len(t)):
            if tt == t[x]:
                tMapper[tt].append(x)

    #structure to hold fluxes
    fluxes = np.zeros(initialFluxes.shape)

    #structure to hold unlabeled fluxes
    c0s = np.zeros(4)

    #define the lower bound for steady state NADH labeling
    lb = np.max([filt["L_lac"].values.mean(),filt["L_malate"].values.mean()])

    #calculate NADH labeling and C0[1] from steady state GAP and G3P labeling
    nadhSS, c0s[1] = calculateSteadyStateNADH(filt["L_gap"].values.mean(), np.array([filt["UL_g3p"].values.mean(), filt["L_g3p_M+1"].values.mean(), filt["L_g3p_M+2"].values.mean()]), vhvds, lb)

    print("ISS NADH labeling: ",nadhSS)

    conc["NADH"] = 1.0

    #define initial conditions
    initialState = [1.0*c for c in [conc["Lactate"],conc["G3P"],conc["Malate"],conc["NADH"]]]

    #fit curve to dhap (gap) data
    weights = [np.min([1,1/np.std(data["UL_gap"].values[tMapper[t[x]]])]) for x in range(len(t))]
    dhap_params = fitSource(t, data["UL_gap"],weights)
    dhap = lambda x: exponetialCurve(x,dhap_params)

    errs = np.zeros(4)

    # lactate
    c0s[0] = C0ConstantMalateLactateNADH(nadhSS, filt["L_lac"].values.mean(),
                                         vhvds["vhvd_nadh_ldh"])

    # malate
    c0s[2] = C0ConstantMalateLactateNADH(nadhSS, filt["L_malate"].values.mean(),
                                         vhvds["vhvd_nadh_mas"])
    #NADH
    c0s[3] = C0ConstantMalateLactateNADH(1-dhap(max(t)), nadhSS,
                                         vhvds["vhvd_gap_gapdh"])

    def updateAndReturnDict(dict, key, val):
        dict[key] = val
        return dict

    def updateAndReturnArray(arr,i,val):
        arr[i] = val
        return arr

    labels1 = ["UL_lac","UL_g3p","UL_malate"]

    # worstCaseSSE = sse(data[labels1].values, integrateLabelingModel(t,np.zeros(4),conc,
    #                     dhap_params,c0s,vhvds,initialState)[:,[0,1,2]])

    fitted = minimize(lambda z: sse(data[labels1].values, integrateLabelingModel(t,updateAndReturnArray(updateAndReturnArray(fluxes,[0,1,2],z[:3]),3, np.sum(z[:3])/(c0s[3]+1)),updateAndReturnDict(conc,"NADH",z[3]),
                        dhap_params,c0s,vhvds,updateAndReturnArray(initialState,3,z[3]))[:,[0,1,2]]) + 1e-5 * np.sum(np.square(z[:3])), x0=np.array(list(initialFluxes[:3])+[conc["NADH"]]),
                      method="Nelder-Mead",
                      options={"fatol": 1e-9}, bounds=[(0, lacE),(0,2*gluUptake),(0,2*gluUptake),(0,None)])

    fluxes[:3] = fitted.x[:3]
    fluxes[3] = np.sum(fitted.x[:3])/(c0s[3]+1)
    conc["NADH"] = fitted.x[3]
    initialState = [1.0*c for c in [conc["Lactate"],conc["G3P"],conc["Malate"],conc["NADH"]]]

    integratedSolution = integrateLabelingModel(t,fluxes,conc,dhap_params,c0s,vhvds,initialState)
    for x,lab in zip([0,1,2],["UL_lac","UL_g3p","UL_malate"]):
        errs[x] = sse(integratedSolution[:,x],data[lab].values)


    data["UL_lac"] = integratedSolution[:,0]
    data["L_lac"] = 1-integratedSolution[:,0]

    data["UL_g3p"] = integratedSolution[:,1]
    data["L_g3p"] = 1-integratedSolution[:,1]

    data["UL_malate"] = integratedSolution[:,2]
    data["L_malate"] = 1-integratedSolution[:,2]

    data["UL_nadh"] = integratedSolution[:,3]
    data["L_nadh"] = 1-integratedSolution[:,3]

    data["UL_gap"] = np.array([dhap(tt) for tt in t])
    data["L_gap"] = 1-np.array([dhap(tt) for tt in t])

    if q:
        q.put(1)

    return fluxes,c0s,data,conc,errs



def generateSyntheticData(ts,noise=.00):
    fluxes = np.random.random(4)
    vhvds = np.random.random(6)
    vhvds = np.power(vhvds,-1)
    #vhvds[2] = 1.0
    labels = ["vhvd_nadh_ldh","vhvd_nadh_mas","vhvd_gap_gapdh","vhvd_nadh_g3ps","vhvd_dhap_g3ps","vhvd_nadh_dhap_g3ps"]
    vhvds = {l:v for l,v in zip(labels,vhvds)}

    unlabeledSources = .3 + .7 * np.random.random(4)

    unlabeledFluxes = np.multiply(unlabeledSources,fluxes)

    fracOfGlycolysis = np.random.random()
    glycolysis = 10 * np.random.random()
    gapdh = glycolysis * fracOfGlycolysis
    fluxes[3] = gapdh
    unlabeledFluxes[3] = gapdh * unlabeledSources[3]
    fluxes[:3] = fluxes[:3] * (gapdh + unlabeledFluxes[3]) / np.sum(fluxes[:3])

    dhap_params = np.random.random(3)
    dhap_params[1] = -1*dhap_params[1] * 3
    dhap_params[[0,2]] = dhap_params[[0,2]] / np.sum(dhap_params[[0,2]])

    conc = np.random.random(4)
    initialState = conc

    switcher = [-1,1]

    conc = {label:con for label,con in zip(["Lactate","G3P","Malate","NADH"],conc)}

    c0s = unlabeledSources

    dynamics = integrateLabelingModel(ts, fluxes, conc, dhap_params, c0s,vhvds, initialState)

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

    result = integrateG3PLabelingModel(ts, fluxes[1],c0s[1], conc["G3P"], nadh, lambda x: exponetialCurve(x,dhap_params), initialState,vhvds)

    df["L_g3p_M+1"] = result[:,1]
    df["L_g3p_M+2"] = result[:,2]

    lacE = fluxes[0] + unlabeledFluxes[0]

    return df,lacE,glycolysis,fluxes,conc,c0s,vhvds


def simulateDataAndInferFlux(ts,numIts,q=None):
    data,lacE,glycolysis,fluxes,conc,c0s,vhvds = generateSyntheticData(ts)
    args = [[data,ts,conc,lacE,glycolysis, vhvds,np.random.random(4)] for _ in range(numIts)]
    startingParams = startConcurrentTask(findFlux,args,1,"finding best fit",len(args),verbose=False)

    startingParams.sort(key=lambda x: np.sum(x[-1]))
    bestParams = startingParams[0]

    if type(q) != type(None):
        q.put(1)
    return fluxes,bestParams
	
def removeBadSol(fluxes,cutoff=0.005,ci=95,target=100):
  
    fluxes = [x for x in fluxes if np.sum(x[-1]) < cutoff]
    if len(fluxes) > target:
      fluxes = rd.sample(fluxes,target)
    fluxes = np.array(fluxes)
    intervalParams = []
    interval = []
    for x in range(fluxes.shape[2]):
      temp = list(fluxes[:,0,x])
      maxi = np.percentile(temp,100-((100-ci)/2),interpolation="nearest")
      mini = np.percentile(temp,(100-ci)/2,interpolation="nearest")
      indOfMax = temp.index(maxi)
      indOfMin = temp.index(mini)
      intervalParams.append([fluxes[indOfMin,:],fluxes[indOfMax,:]])
      interval.append([mini,maxi]) 
      

    return interval,intervalParams,fluxes

def exponetialCurve(t,params):
    return params[0] * np.exp(params[1] * t) + params[2]


def fitSource(t,l,weights=None):
    good = False
    errorFunc = lambda x: sse(exponetialCurve(t,x),l,weights)
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


def runMonteCarlo(data, t, conc, exc, gluUp,vhvds, initialParams=np.random.rand(7, 1), numIts=100,numCores=4):
    # define monte carlo datasets

    datasets = startConcurrentTask(resampledata,[[data,t] for _ in range(numIts)],numCores,"resampling datasets",numIts)
    args = [[dataset,t,conc,exc,gluUp,vhvds,initialParams] for dataset in datasets]

    result = startConcurrentTask(findFlux,args,numCores,"running monte carlo",numIts)

    fluxes = [[x[0],x[1],x[-1]] for x in result if type(x) != type(-1)]

    print(len(fluxes),"successful iterations complete")

    fluxes = np.array(fluxes)

    return fluxes






