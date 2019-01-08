# -*- coding: utf-8 -*-

import numpy as np
import functions as f

def algo(problem,problemConverted):

    (m,adjN) = np.shape(problemConverted['A'])
    guessPt = f.getGuessPt(adjN,m)
    tol = 1e-6
    err = 1e2
    count = 0
    k = 0
    while err > tol and count<100:    
        deltaVarAff = f.getDeltaAff(problemConverted,guessPt,adjN,m)
        alphaAff = f.getAlphaAff(guessPt,deltaVarAff,adjN,m)
        us = f.getSigmaMu(guessPt,alphaAff[-1],adjN,m)
        deltaVar = f.getDelta(problemConverted,guessPt,adjN,m,us,deltaVarAff)
        eta = f.getEta(k)
        alpha = f.getAlpha(guessPt,deltaVar,adjN,m,alphaAff,eta)
        guessPtNew = f.updateGuess(guessPt,deltaVar,alpha,adjN,m)
        err = f.getDiff(guessPt,guessPtNew)
        k = k+1
        count = count+1
        guessPt = guessPtNew
    
    sol = f.getSol(guessPt,int((adjN-m)/2)) 
    val = f.getVal(problem,sol)
    return (sol,val)