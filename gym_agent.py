from abc import ABCMeta, abstractmethod


class gym_agent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, inout):
        pass

    @abstractmethod
    def step(self, observation, reward, done):
        pass
