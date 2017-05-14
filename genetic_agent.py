from gym_agent import gym_agent

from deap import base, creator, tools
from abc import ABCMeta, abstractmethod


class genetic_agent(gym_agent,metaclass = ABCMeta):
    def __init__(self, inout, individual_class, population=list):
        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", individual_class, fitness=creator.Fitness)

        toolbox = base.Toolbox()  # make functions
        toolbox.register("attr_item",)# gene generate routine
        toolbox.register("individual", tools.initRepeat, creator.Individual)# individual generate routine
        toolbox.register("population", tools.initRepeat, population, toolbox.individual)

        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", self.mate)
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", self.select)

    @abstractmethod
    def evaluate(self,individual):
        pass

    @abstractmethod
    def mate(self,ind1,ind2):
        pass

    @abstractmethod
    def mutate(self,individual):
        pass

    @abstractmethod
    def select(self,):
        pass
