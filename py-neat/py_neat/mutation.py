import abc
from typing import List, Set
from core import NeuronType
from genes import NeuronGene, ConnectionGene

from core import MutationOperation, EvolutionParameters, InnovationsCollector, Organism, FitnessCollector, Gene, \
    NeuronType, GeneType


class BaseMutationOperation(MutationOperation, abc.ABC):
    mutation_probability: float = 0.0

    def __init__(self, probability: float):
        self.mutation_probability = probability

    def mutate(self, par: 'EvolutionParameters', innovations: 'InnovationsCollector', fitnesses: 'FitnessCollector',
               organisms: List[Organism], generation: int) -> int:
        mutation_count: int = 0

        for o in organisms:
            genes_to_add: Set[Gene] = set()
            genes_to_remove: Set[Gene] = set()

            if self._should_mutate(par):
                if self._mutate(par, innovations, fitnesses, o, genes_to_add, genes_to_remove, generation):
                    mutation_count += 1

                for g in genes_to_add:
                    innovations.add_innovation(g)
                    o.genes.append(o)

                for g in genes_to_remove:
                    o.genes.remove(g)

        return mutation_count

    def _should_mutate(self, par: EvolutionParameters) -> bool:
        return par.random.random() < self.mutation_probability

    def num_mutations_to_perform(self, par: EvolutionParameters, max: int):
        total = 0

        for i in range(0, max):
            if self._should_mutate(par):
                total += 1

        return total

    @abc.abstractmethod
    def _mutate(self, par, innovations, fitnesses, o, genes_to_add, genes_to_remove, generation):
        pass


class ActivationResponse(BaseMutationOperation):
    __DEFAULT_MAX_ACTIVATION_PERTURBATION__ = 0.1

    def __init__(self, probability: float = 0.1):
        super(ActivationResponse, self).__init__(probability)
        self.max_activation_pertubation = self.__DEFAULT_MAX_ACTIVATION_PERTURBATION__

    def _mutate(self, par: EvolutionParameters, innovations: InnovationsCollector, fitnesses: FitnessCollector,
                o: Organism, genes_to_add: Set[Gene], genes_to_remove: Set[Gene], generation: int):
        genes: List[Gene] = o.get_neurons(NeuronType.HIDDEN)

        if len(genes) == 0:
            return False

        random_neuron: NeuronGene = genes[par.random.randint(0, len(genes))]
        # we want a number between -1 and 1, but random gives us a number between 0..1, so call it twice and subtract
        random_neuron.activation_response = random_neuron.activation_response + ((
                                                                                         par.random.random() - par.random.random()) * self.max_activation_pertubation)
        return True


class AddConnection(BaseMutationOperation):
    __DEFAULT_ADD_CONNECTION_PROBABILITY__: float = 0.07
    __MAX_ATTEMPTS__ = 15

    def __init__(self, probability: float = __DEFAULT_ADD_CONNECTION_PROBABILITY__):
        super(AddConnection, self).__init__(probability)

    def _mutate(self, par: EvolutionParameters, innovations: InnovationsCollector, fitnesses: FitnessCollector,
                o: Organism, genes_to_add: Set[Gene], genes_to_remove: Set[Gene], generation: int) -> bool:

        neurons: List[NeuronGene] = filter(lambda x: x.gene_type == GeneType.NEURON, o.genes)

        for loop_count in range(0, AddConnection.__MAX_ATTEMPTS__):
            start: NeuronGene = neurons[par.random.randint(0, len(neurons))]
            end: NeuronGene = neurons[par.random.randint(0, len(neurons))]

            if AddConnection.is_valid_connection(o, genes_to_add, genes_to_remove):
                gene: ConnectionGene = ConnectionGene(par, start.innovation_id, end.innovation_id, par.random.random(),
                                                      True)
                genes_to_add.add(gene)

                return True

        return False

    @staticmethod
    def is_valid_connection(o: Organism, genes_to_add: Set[Gene], genes_to_remove: Set[Gene], start_gene: NeuronGene,
                            end_gene: NeuronGene) -> bool:
        # make sure we aren't trying to connect to ourselves
        # TODO: this is probably valid, but should be a configuration option
        if start_gene == end_gene:
            return False

        # go through and make sure we aren't doubling-up the connections in the existing organism's structure
        for g in filter(lambda x: x.gene_type == GeneType.CONNECTION, o.genes):
            if g.origin_id == start_gene.innovation_id and g.endpoint_id == end_gene.innovation_id:
                return False

            if g.endpoint_id == start_gene.innovation_id and g.origin_id == end_gene.innovation_id:
                return False

        if start_gene.neuron_type == NeuronType.INPUT and end_gene.neuron_type == NeuronType.INPUT:
            return False

        if start_gene.neuron_type == NeuronType.OUTPUT and end_gene.neuron_type == NeuronType.OUTPUT:
            return False

        for g in list(genes_to_add) + list(genes_to_remove):
            if g.gene_type == GeneType.NEURON:
                continue

            if g.origin_id == start_gene.innovation_id or g.endpoint_id == end_gene.innovation_id:
                return False

        return True


class AddNeuron(BaseMutationOperation):
    __DEFAULT_ADD_NEURON_PROBABILITY__: float = 0.04

    def __init__(self, probability: float = __DEFAULT_ADD_NEURON_PROBABILITY__):
        super(AddNeuron, self).__init__(probability)

    def _mutate(self, par: EvolutionParameters, innovations: InnovationsCollector, fitnesses: FitnessCollector,
                o: Organism, genes_to_add: Set[Gene], genes_to_remove: Set[Gene], generation: int) -> bool:
        connections: List[Gene] = list(filter(lambda x: x.gene_type == GeneType.CONNECTION, o.genes))

        random_connection: ConnectionGene = connections[par.random.randint(0, len(connections))]

        # adding a new gene requires taking an existing connection, breaking it and adding a neuron inbetween the two
        start_neuron: NeuronGene = next(filter(lambda x: x.innovation_id == random_connection.origin_id, o.genes), None)
        end_neuron: NeuronGene = next(filter(lambda x: x.innovation_id == random_connection.endpoint_id, o.genes), None)

        new_neuron: NeuronGene = NeuronGene(par.next_innovation_id(), NeuronType.HIDDEN, par.random.random())
        left_half: ConnectionGene = ConnectionGene(par.next_innovation_id(), start_neuron.innovation_id,
                                                   new_neuron.innovation_id, par.random.random())
        right_half: ConnectionGene = ConnectionGene(par.next_innovation_id(), new_neuron.innovation_id,
                                                    end_neuron.innovation_id, par.random.random())

        # disable the original connection as it's been replaced by the two above
        random_connection.enabled = False

        genes_to_add.add(new_neuron)
        genes_to_add.add(left_half)
        genes_to_add.add(right_half)

        return True


class AdjustWeight(BaseMutationOperation):
    __DEFAULT_ADJUST_WEIGHT_PROBABILITY__: float = 0.1
    __MAXIMUM_WEIGHT_PERTUBATION__: float = 0.5

    def __init__(self, probability: float = __DEFAULT_ADJUST_WEIGHT_PROBABILITY__, maxpertubation: float = __MAXIMUM_WEIGHT_PERTUBATION__):
        super(AdjustWeight, self).__init__(probability)
        self.maximum_weight_pertubation = maxpertubation

    def _mutate(self, par: EvolutionParameters, innovations: InnovationsCollector, fitnesses: FitnessCollector,
                o: Organism, genes_to_add: Set[Gene], genes_to_remove: Set[Gene], generation: int) -> bool:
        connections: List[Gene] = list(filter(lambda x: x.gene_type == GeneType.CONNECTION, o.genes))
        random_connection: ConnectionGene = connections[par.random.randint(0, len(connections))]

        random_connection.weight = random_connection.weight + ((par.random.random() - par.random.random()) * self.maximum_weight_pertubation)

        return True
