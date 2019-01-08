# -*- coding: utf-8 -*-

import numpy as np

def problemGenerator(n , m):
    #The problem is of the form
    # min c^Tx
    # s.t. Ax<=b
    #constraint list
    x = np.random.rand(n)
    problem = {}
    problem['A'] = np.array(np.random.rand(m,n))
    problem['b'] = problem['A']@x +\
    np.concatenate(
            (np.random.rand(int(np.floor(m/2))),
             -np.random.rand(int(np.ceil(m/2))))
            )
    problem['A'][int(np.ceil(m/2)):] = -problem['A'][int(np.ceil(m/2)):]
    problem['b'][int(np.ceil(m/2)):] = -problem['b'][int(np.ceil(m/2)):]
    
    #objective function
    problem['c'] = np.array(np.random.randn(n))
    return problem

#check this function
def problemConverter(problem):
    #define number of constraints
    m = len(problem['A'][:,0])
    
    #add the extra variables for only positive
    problem['c'] = np.repeat(problem['c'],2)
    problem['c'][1::2] = -1*problem['c'][1::2]
    
    problem['A'] = np.repeat(problem['A'],2,1)
    problem['A'][:,1::2] = -1*problem['A'][:,1::2]
    
    #add the slack variables
    problem['c'] = np.pad(problem['c'],(0,m),'constant')
    problem['A'] = np.concatenate((problem['A'],np.eye(m)),axis=1)
    return problem


def getGuessPt(n,m):
    #2*variables+slack,lagrange multipliers, dual variables
    guessPt = np.random.rand(n+m+n)
    return guessPt

def getDeltaAff(problem,guessPt,n,m):
    S = np.diagflat(guessPt[n+m:])
    X = np.diagflat(guessPt[:n])
    A = problem['A']
    I = np.eye(n)
    kktMat = np.block([
    [np.zeros((n,n)),np.transpose(A),I],
    [A,np.zeros((m,m)),np.zeros((m,n))],
    [S,np.zeros((n,m)),X]
    ])
    
    rc = np.transpose(A) @ guessPt[n:n+m] + guessPt[n+m:] - problem['c']
    rb = A @ guessPt[:n]-problem['b']
    XSe = X @ S @ np.ones(n) 
    
    b = np.block([
            -rc,
            -rb,
            -XSe
            ])
    kktMatInv = np.linalg.inv(kktMat)
    deltaVarAff = kktMatInv @ b
    return deltaVarAff

def getAlphaAff(guessPt,deltaVar,n,m):
    x = guessPt[:n]
    deltaX = deltaVar[:n]
    s = guessPt[n+m:]
    deltaS = deltaVar[n+m:]
    
    alphaPriAff = 1
    alphaDualAff = 1
    for i in range(n):
        if deltaX[i]<0:
            alphaPriAff = min(alphaPriAff,-x[i]/deltaX[i])
        if deltaS[i]<0:
            alphaDualAff = min(alphaDualAff,-s[i]/deltaS[i])
    uAff = np.dot(x + alphaPriAff*deltaX,s+alphaDualAff*deltaS)/n
    alphaAff = np.array([alphaPriAff,alphaDualAff,uAff])
    return alphaAff

def getSigmaMu(guessPt,uAff,n,m):
    u = np.dot(guessPt[:n],guessPt[n+m:])
    test = (uAff/u)**3
    us = np.array([u,test])
    return us


def getDelta(problem,guessPt,n,m,us,deltaVarAff):
    S = np.diagflat(guessPt[n+m:])
    X = np.diagflat(guessPt[:n])
    A = problem['A']
    I = np.eye(n)
    kktMat = np.block([
    [np.zeros((n,n)),np.transpose(A),I],
    [A,np.zeros((m,m)),np.zeros((m,n))],
    [S,np.zeros((n,m)),X]
    ])
    
    rc = np.transpose(A) @ guessPt[n:n+m] + guessPt[n+m:] - problem['c']
    rb = A @ guessPt[:n]-problem['b']
    XSe = X @ S @ np.ones(n) 
    
    b = np.block([
            -rc,
            -rb,
            -XSe +(np.prod(us) - np.dot(deltaVarAff[:n],deltaVarAff[n+m:]))*np.ones(n)
            ])
    kktMatInv = np.linalg.inv(kktMat)
    deltaVar = kktMatInv @ b
    return deltaVar
def getEta(k):
    #simple functions that starts at 0.9 and approaches 1
    eta = (k+6.53)/np.sqrt((k+6.53)**2+10)
    return eta

def getAlpha(guessPt,deltaVar,n,m,alphaAff,eta):
    x = guessPt[:n]
    deltaX = deltaVar[:n]
    s = guessPt[n+m:]
    deltaS = deltaVar[n+m:]
    
    alphaPriMax = 1e5
    alphaDualMax = 1e5
    for i in range(n):
        if deltaX[i]<0:
            alphaPriMax = min(alphaPriMax,-x[i]/deltaX[i])
        if deltaS[i]<0:
            alphaDualMax = min(alphaDualMax,-s[i]/deltaS[i])
    
    alphaPri = min(1,alphaPriMax*eta)
    alphaDual = min(1,alphaDualMax*eta)
    alpha = np.array([alphaPri,alphaDual])
    return alpha

def updateGuess(guessPt,deltaVar,alpha,n,m):
    x = guessPt[:n]+alpha[0]*deltaVar[:n]
    lambdaS = guessPt[n:]+alpha[1]*deltaVar[n:]
    guessPtNew = np.concatenate((x,lambdaS))
    return guessPtNew

def getDiff(guessPt,guessPtNew):
    err = np.linalg.norm(guessPt-guessPtNew)
    return err
    
def getSol(guessPt,n):
    sol=[]
    for i in range(n):
        sol.append(guessPt[2*i]-guessPt[2*i+1])
    return sol
def getVal(problem,sol):
    val = problem['c']@sol
    return val