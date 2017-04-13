#!/usr/bin/env python3

from deap import base,creator,tools
import numpy as np

creator.create("FitnessMin",base.Fitness, weights=(1.0,))#init adapt class
#creator.create("Individual", list, fitness=creator.FitnessMin)#?
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)#?

toolbox = base.Toolbox()#make functions
toolbox.register("attr_bool", np.random.randint, 2)#define attr_bool = random.randint(0,1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)#
toolbox.register("population", tools.initRepeat, list, toolbox.individual)#

def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)#
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == '__main__':
    pop = toolbox.population(n=300)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40#mate probability, mutate probability, generation times

    print('start of evolution')

    #evaluate parents
    fitnesses = list(map(toolbox.evaluate,pop))
    for ind,fit in zip(pop,fitnesses):
        ind.fitness.value = fit

    #start generation
    for g in range(NGEN):
        print("-- Generation",g,"--")

        #select new generations
        offspring = list(map(toolbox.clone, toolbox.select(pop,len(pop))))

        #mate&mutate
        for child1,child2 in zip(offspring[::2],offspring[1::2]):
            if np.random.random() < CXPB:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if np.random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        #calc fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        #next gen <- offspring
        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits)/length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2/length - mean**2)**0.5

        print("\tEvaluated {} individuals\t".format(len(invalid_ind)),"min:{}\tmax:{}\tavg:{:.5f}\tstd:{:.5f}".format(min(fits),max(fits),mean,std))

    print("--end--")

    best_ind = tools.selBest(pop,1)[0]
    print("best : {},{}".format(best_ind,best_ind.fitness.values))
