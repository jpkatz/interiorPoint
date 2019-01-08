# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
def plotProblem(problem):
    m = len(problem['A'])
    total=5
    x = np.linspace(-10,10,total)
    y=np.empty((m,total))
    for row in range(0,m):
        y[row,:] = (problem['b'][row] - problem['A'][row][0]*x)/problem['A'][row][1]
        
    for idx in range(0,m):
        plt.plot(x,y[idx,:])
    plt.xlim((-4,4))

def plotMat(times,varCount,conCount):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(times,interpolation='nearest',
                    extent=[min(varCount),max(varCount),min(conCount),max(conCount)],
                    aspect='auto',
                    origin='lower')
    fig.colorbar(im)
    plt.show()
    
def plotHisto(times,varCount,conCount):
    test = np.array(times)
    heatmap, xedges, yedges = np.histogram2d(sorted(varCount*len(varCount)),conCount*len(varCount), bins=(len(varCount)/2,len(varCount)/2),weights=test.ravel())
    extent = [min(varCount),max(varCount),min(conCount),max(conCount)]
    # Plot heatmap
    plt.clf()
    plt.ylabel('y')
    plt.xlabel('x')
    im=plt.imshow(heatmap,extent=extent,origin='low')
    plt.colorbar(im)
    plt.show()