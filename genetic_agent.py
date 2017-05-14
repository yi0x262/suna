from gym_agent import gym_agent

from deap import base, creator, tools


class genetic_agent(creator, gym_agent):
    def __init__(self, inout, individual_class):
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", individual_class, fitness=creator)
