from enum import Enum
from typing import Optional
import math


# Enum for different terrain types
class TerrainType(Enum):
    EMPTY = 0
    WALL = 1
    COLONY = 2
    FOOD = 3


# Enum for directions
class Direction(Enum):
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7

    @staticmethod
    def get_delta(direction):
        direction_value = (
            direction.value if isinstance(direction, Direction) else direction
        )
        deltas = [
            (0, -1),  # North
            (1, -1),  # Northeast
            (1, 0),  # East
            (1, 1),  # Southeast
            (0, 1),  # South
            (-1, 1),  # Southwest
            (-1, 0),  # West
            (-1, -1),  # Northwest
        ]
        if 0 <= direction_value < len(deltas):
            return deltas[direction_value]
        return (0, 0)

    @staticmethod
    def get_left(direction):
        return Direction((direction.value - 1) % 8)

    @staticmethod
    def get_right(direction):
        return Direction((direction.value + 1) % 8)


# Enum for ant actions
class AntAction(Enum):
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    PICK_UP_FOOD = 3
    DROP_FOOD = 4
    DEPOSIT_HOME_PHEROMONE = 5
    DEPOSIT_FOOD_PHEROMONE = 6
    NO_ACTION = 7


# Class for perception information
class AntPerception:
    """Class representing what an ant can perceive from its environment"""

    def __init__(self):
        self.visible_cells = {}
        self.food_pheromone = {}
        self.home_pheromone = {}
        self.nearby_ants = []

        # Ant-specific properties
        self.has_food = False
        self.direction = None
        self.home_pheromone_level = None
        self.food_pheromone_level = None
        self.pheromone_decrease_rate = None
        self.food_collected = 0
        self.steps_taken = 0
        self.ant_id = None

    def can_see_food(self) -> bool:
        return TerrainType.FOOD in [cell for cell in self.visible_cells.values()]

    def can_see_colony(self) -> bool:
        return TerrainType.COLONY in [
            cell for cell in self.visible_cells.values()
        ]

    def get_food_direction(self) -> Optional[int]:
        min_dist = float("inf")
        best_dir = None

        for (dx, dy), cell_type in self.visible_cells.items():
            if cell_type == TerrainType.FOOD:
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
                    best_dir = self._get_direction_from_delta(dx, dy)

        return best_dir

    def get_colony_direction(self) -> Optional[int]:
        min_dist = float("inf")
        best_dir = None

        for (dx, dy), cell_type in self.visible_cells.items():
            if cell_type == TerrainType.COLONY:
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
                    best_dir = self._get_direction_from_delta(dx, dy)

        return best_dir

    def _get_direction_from_delta(self, dx: int, dy: int) -> int:
        if dx == 0 and dy < 0:
            return Direction.NORTH.value
        elif dx > 0 and dy < 0:
            return Direction.NORTHEAST.value
        elif dx > 0 and dy == 0:
            return Direction.EAST.value
        elif dx > 0 and dy > 0:
            return Direction.SOUTHEAST.value
        elif dx == 0 and dy > 0:
            return Direction.SOUTH.value
        elif dx < 0 and dy > 0:
            return Direction.SOUTHWEST.value
        elif dx < 0 and dy == 0:
            return Direction.WEST.value
        elif dx < 0 and dy < 0:
            return Direction.NORTHWEST.value
        return Direction.NORTH.value
