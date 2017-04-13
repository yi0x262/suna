#!/usr/bin/env python3
class Novelty_Map(list):
    def __init__(self,max_population):
        self.max_population = max_population
        super().__init__()
    def append(self,arg):
        """if it has no similar element -> append it"""
        if len(self) < self.max_population:
            super().append(arg)
            return
        #if min(map(self._minimum_distance,self)) < self._minimum_distance(arg):
        min_dist = self._minimum_distance(arg)
        for i in self:
            if self._minimum_distance(i) < min_dist:
                super().remove(i)
                super().append(arg)
                break

    def _minimum_distance(self,arg):
        """arg.dist2 should return number > 0"""
        return min([abs(arg-e) for e in self if not e is arg])

if __name__ == '__main__':
    nm = Novelty_Map(3)
    nm.extend(list(range(10000)))
    nm.append(10000)
    print(nm)
