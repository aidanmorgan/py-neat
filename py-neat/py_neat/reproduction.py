import abc
from typing import List

from core import ReproductionOperation, Organism, InnovationsCollector, FitnessCollector, EvolutionParameters, Specie


class BaseReproductionOperation(ReproductionOperation, abc.ABC):
    __DEFAULT_BIAS_FITTER_ORGANISM__: bool = True

    def __init__(self, probability: float, biasfitter: bool = __DEFAULT_BIAS_FITTER_ORGANISM__):
        self.probability = probability
        self.bias_fitter_organism = biasfitter

    def reproduce(self, par: EvolutionParameters, innovations: InnovationsCollector, scores: FitnessCollector,
                  count: int, species: List[Specie], organisms: List[Organism], generation: int):
        children:List[Organism] = list()
        size:int = len(organisms)

        while len(children) < size + count:
            if self.bias_fitter_organism:
                self.reproduce_with_bias()
            else:
                self.reproduce_without_bias()


    def reproduce_with_bias(self):
        overall_fitness: float = sum(map(lambda x: scores.score_for_organism(x, generation), organisms))
        pass

    def reproduce_without_bias(self):
        pass



class CloneReproductionOperation(BaseReproductionOperation):
    __DEFAULT_OPERATION_PROBABILITY__: float = 0.0

    def __init__(self,
                 probability: float = __DEFAULT_OPERATION_PROBABILITY__,
                 biasfitter: bool = BaseReproductionOperation.__DEFAULT_BIAS_FITTER_ORGANISM__):
        super(CloneReproductionOperation, self).__init__(probability, biasfitter)

    def reproduce(self, par: EvolutionParameters, innovations: InnovationsCollector, scores: FitnessCollector,
                  count: int, species: List[Specie], parents: List[Organism], generation: int):

        pass


class CrossoverReproductionOperation(BaseReproductionOperation):

    def reproduce(self, par: 'EvolutionParameters', innovations: 'InnovationsCollector', scores: 'FitnessCollector',
                  count: int, species: List['Specie'], organisms: List['Organism'], generation: int):
        pass
