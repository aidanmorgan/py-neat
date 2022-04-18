import abc
from abc import abstractmethod, ABC
from collections import namedtuple
from enum import Enum
from typing import Callable, List, Dict, Type

from neurolab.core import Net
import random as r


class NeuronType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class GeneType(Enum):
    NEURON = 0
    CONNECTION = 1


class EvolutionEvent(Enum):
    START_GENERATION = 0
    END_GENERATION = 1

    START_SELECTION = 2
    END_SELECTION = 4

    START_REPRODUCTION = 8
    END_REPRODUCTION = 16

    START_MUTATION = 32
    END_MUTATION = 64

    START_FITNESS_CALCULATION = 128
    END_FITNESS_CALCULATION = 256

    START_SPECIATION = 512
    END_SPECIATION = 1024


class Innovation(abc.ABC):
    innovation_id: int

    @abstractmethod
    def copy(self):
        pass


class Organism(Innovation, ABC):
    specie: 'Specie'
    genes: List['Gene'] = list()

    def __init__(self):
        pass


class FitnessFunction(abc.ABC):
    @abstractmethod
    def evaluate(self, net: Net) -> float:
        pass


class OrganismSelector(abc.ABC):
    @abstractmethod
    def select_for_next_generation(self):
        pass


class SpecieSelector(abc.ABC):
    @abstractmethod
    def select_for_next_generation(self):
        pass


class ReproductionOperation(abc.ABC):
    @abstractmethod
    def reproduce(self, par: 'EvolutionParameters', innovations: 'InnovationsCollector', scores: 'FitnessCollector',
                  count: int, species: List['Specie'], parents: List[Organism], generation: int):
        pass


class MutationOperation(abc.ABC):
    @abstractmethod
    def mutate(self, par: 'EvolutionParameters', innovations: 'InnovationsCollector', fitnesses: 'FitnessCollector',
               organisms: List[Organism], generation: int) -> int:
        pass


class Speciator(abc.ABC):
    @abstractmethod
    def speciate(self, params, species: List['Specie'], scores: 'FitnessCollector', chromosomes: List[Organism]):
        pass

    @abstractmethod
    def compare(self, one: Organism, two: Organism):
        pass


class Gene(Innovation, abc.ABC):
    gene_type: GeneType
    enabled: bool = True


class Generation:
    organisms: List[Organism] = list()
    generation: int


class Specie(Innovation):
    organisms: List[Organism] = list()
    fitness_scores: 'FitnessCollector'
    parameters: 'EvolutionParameters'

    representative_organism: Organism = None
    alive: bool = True

    def __init__(self, p: 'EvolutionParameters', organisms: List[Organism]):
        self.innovation_id = p.next_innovation_id()
        self.parameters = p
        self.organisms = organisms

        if len(organisms) > 0:
            self.representative_organism = organisms[0]

    def cull(self, organisms: List[Organism]):
        for o in organisms:
            self.organisms.remove(o)

    def kill(self):
        self.alive = False

    def copy(self) -> 'Specie':
        clone: List[Organism] = list(map(lambda x: x.copy(), self.organisms))
        return Specie(self.parameters, clone)


FitnessKey = namedtuple('FitnessKey', 'generation innovation_id')


class FitnessCollector:
    fitness_scores: Dict[FitnessKey, float] = dict()

    def add_organism(self, o: Organism, generation: int, score: float):
        key: FitnessKey = FitnessKey(generation, o.innovation_id)

        self.fitness_scores[key] = score

    def score_for_organism(self, o: Organism, generation:int = None):
        if generation is None:
            keys: List[FitnessKey] = filter(lambda x: x.innovation_id == o.innovation_id, self.fitness_scores.keys())

            if len(keys) > 0:
                keys.sort(key=lambda x: x.generation, reverse=True)
                return self.fitness_scores[keys[0]]
            else:
                return None
        else:
            key:FitnessKey = FitnessKey(generation, o.innovation_id)

            if key in self.fitness_scores:
                return self.fitness_scores[key]

            return None


InnovationsKey = namedtuple('InnovationsKey', 'type innovation_id')


class InnovationsCollector:
    innovations: Dict[InnovationsKey, Innovation] = dict()

    def add_innovation(self, innov: Type[Innovation]):
        key: InnovationsKey = InnovationsKey(type(innov), innov.innovation_id)
        self.innovations[key] = innov


class EvolutionParameters:
    population_size: int = 1
    fitness_function_factory: Callable[[], FitnessFunction]
    next_innovation_id: Callable[[], int]

    organism_selector: OrganismSelector
    speciator: Speciator

    reproduction_operators: List[ReproductionOperation] = list()

    import mutations
    mutation_operators: List[MutationOperation] = mutations.DEFAULT_MUTATIONS
    random = r

    network_builder: Callable[[Organism], Net]
    speciator: Callable[[Specie], bool]

    termination_condition: Callable[['EvolutionParameters', int, Generation, Dict[int, float]], bool]
