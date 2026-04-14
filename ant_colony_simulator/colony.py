import random
from typing import List, Dict
from environment import Direction
from ant import Ant, AntStrategy


class Colony:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.ants = []
        self.total_food_collected = 0

    def add_ant(self, ant: Ant) -> None:
        """Add an ant to the colony"""
        self.ants.append(ant)

    def create_ants(self, count: int, strategy: AntStrategy) -> List[Ant]:
        """Create ants with the given strategy"""
        new_ants = []
        for _ in range(count):
            direction = random.choice(list(Direction))
            ant = Ant(self.x, self.y, direction, strategy)
            self.ants.append(ant)
            new_ants.append(ant)
        return new_ants

    def update_food_count(self) -> None:
        """Update total food collected by the colony"""
        self.total_food_collected = sum(ant.food_collected for ant in self.ants)

    def get_stats(self) -> Dict:
        """Get colony statistics"""
        self.update_food_count()
        return {
            "total_ants": len(self.ants),
            "food_collected": self.total_food_collected,
            "avg_steps_per_food": sum(ant.steps_taken for ant in self.ants)
            / max(1, self.total_food_collected),
        }
