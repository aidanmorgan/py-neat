from core import Gene, EvolutionParameters, NeuronType, GeneType


class NeuronGene(Gene):

    neuron_type: NeuronType
    activation_response: float

    @classmethod
    def create(cls, par: EvolutionParameters, t: NeuronType, activation:float, enabled: bool = True):
        return NeuronGene(par.next_innovation_id(), t, activation, enabled)

    def __init__(self, innovationid: int, t: NeuronType, activation:float, enabled: bool = True):
        self.innovation_id = innovationid
        self.neuron_type = t
        self.activation_response = activation
        self.enabled = enabled
        self.gene_type = GeneType.NEURON

    def copy(self) -> 'NeuronGene':
        return NeuronGene(self.innovation_id, self.neuron_type, self.activation_response, self.enabled)



class ConnectionGene(Gene):
    origin_id: int
    endpoint_id: int
    weight: float

    @classmethod
    def create(cls, par: EvolutionParameters, start: int, end: int, weight: float, enabled: bool = True):
        return ConnectionGene(par.next_innovation_id(), start, end, weight, enabled)

    def __init__(self, innovationid: int, start: int, end: int, weight: float, enabled: bool = True):
        self.innovation_id = innovationid
        self.origin_id = start
        self.endpoint_id = end
        self.weight = weight
        self.enabled = enabled
        self.gene_type = GeneType.CONNETION

    def copy(self, param: 'EvolutionParameters') -> 'ConnectionGene':
        clone: ConnectionGene = ConnectionGene(self.innovation_id, self.origin_id, self.endpoint_id, self.weight, self.enabled)
        return clone
