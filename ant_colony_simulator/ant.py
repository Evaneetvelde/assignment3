from abc import ABC, abstractmethod
from common import Direction, AntPerception, AntAction


# Strategy interface for ant behavior
class AntStrategy(ABC):
    @abstractmethod
    def decide_action(self, perception: AntPerception) -> AntAction:
        """Decide the action of an ant based on its perception"""
        pass

    def get_name(self) -> str:
        """Get strategy name"""
        return self.__class__.__name__


# Ant class with possible actions
class Ant:
    def __init__(
        self,
        x: int,
        y: int,
        direction: Direction,
        strategy: AntStrategy,
        ant_id: int = None,
    ):
        self.x = x
        self.y = y
        self.direction = direction
        self.strategy = strategy
        self.has_food = False
        self.home_pheromone = 100.0
        self.food_pheromone = 100.0
        self.pheromone_decrease_rate = 0.995
        self.vision_range = 3  # How far the ant can see
        self.vision_angle = 120  # Field of view angle in degrees (total angle)
        self.food_collected = 0
        self.steps_taken = 0
        self.id = ant_id

    def set_strategy(self, strategy: AntStrategy) -> None:
        self.strategy = strategy

    def decide_action(self, perception: AntPerception) -> AntAction:
        if self.strategy:
            self.steps_taken += 1
            return self.strategy.decide_action(perception)
        return AntAction.NO_ACTION

    def turn_left(self) -> None:
        self.direction = Direction.get_left(self.direction)

    def turn_right(self) -> None:
        self.direction = Direction.get_right(self.direction)

    def move_forward(self, success: bool) -> None:
        if success:
            dx, dy = Direction.get_delta(self.direction)
            self.x += dx
            self.y += dy

    def pick_up_food(self, success: bool) -> None:
        if success:
            self.has_food = True

    def drop_food(self, success: bool) -> None:
        if success and self.has_food:
            self.has_food = False
            self.food_collected += 1
            # Reset pheromone levels
            self.home_pheromone = 100.0
            self.food_pheromone = 100.0

    def deposit_pheromone(self) -> float:
        if self.has_food:
            amount = self.food_pheromone
            self.food_pheromone *= self.pheromone_decrease_rate
            return amount
        else:
            amount = self.home_pheromone
            self.home_pheromone *= self.pheromone_decrease_rate
            return amount
