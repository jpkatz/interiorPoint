# -*- coding: utf-8 -*-

import algo,functions
import time

#building list of number of variables and constraints
varCount = [x for x in range(1,201) if not x%10]
conCount = varCount
#initializes length of the list for time to solve
timeStore = [[] for x in range(len(varCount))]
plot = 1;

for i in range(len(varCount)):
    j = i+1
    timeStore[i]=[0]*j+timeStore[i]
    while j<len(varCount):
        #build a random problem and convert it to proper form
        problem = functions.problemGenerator(varCount[i],conCount[j]) 
        problemConverted = problem.copy()
        problemConverted = functions.problemConverter(problemConverted)
        #try catch in case problem not solvable
        try:
            timeStart = time.time()
            solution = algo.algo(problem,problemConverted)
            timeFinal = time.time()
            deltaTime = timeFinal-timeStart
            timeStore[i].append(deltaTime)
            j=j+1
        except KeyboardInterrupt:
            raise
        except:
            print('Failure')
        print(i,j)

if plot:
    import plotProblem
    #plot of length of times to solve for different n and m 
    plotProblem.plotMat(timeStore,varCount,conCount)
    #pseudo heat map since it is clear how the algorithm will scale since
    #only m>n combinations are chosen
    plotProblem.plotHisto(timeStore,varCount,conCount)
