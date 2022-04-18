from core import OrganismSelector


class NaturalSelectionSelector(OrganismSelector):
    __DEFAULT_SURVIVAL_RATIO__: float = 0.0
    __DEFAULT_MAX_GENERATIONS_WITH_NO_IMPROVEMENT__: int = 20
    __DEFAULT_KILL_UNPRODUCTIVE_SPECIES__: bool = False
    __DEFAULT_ELITISM_ENABLED__: bool = True

    def __init__(self, survivalratio:float = __DEFAULT_SURVIVAL_RATIO__, maxgenerationsnoimprovement:int = __DEFAULT_MAX_GENERATIONS_WITH_NO_IMPROVEMENT__,
                 killunproductivespecies:bool = __DEFAULT_KILL_UNPRODUCTIVE_SPECIES__, elitismenabled: bool = __DEFAULT_ELITISM_ENABLED__):
        self.survival_ratio = survivalratio
        self.maximum_generations_with_no_improvement = maxgenerationsnoimprovement
        self.is_kill_unproductive_species = killunproductivespecies
        self.is_elitism_enabled = elitismenabled

    def select_for_next_generation(self):
        if self.is_elitism_enabled:
            pass

