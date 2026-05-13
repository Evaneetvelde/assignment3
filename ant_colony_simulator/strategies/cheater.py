from environment import TerrainType, AntPerception
from ant import AntAction, AntStrategy
from common import Direction

import heapq
import math
import random


DIRECTION_ORDER = [
    Direction.NORTH,
    Direction.NORTHEAST,
    Direction.EAST,
    Direction.SOUTHEAST,
    Direction.SOUTH,
    Direction.SOUTHWEST,
    Direction.WEST,
    Direction.NORTHWEST,
]

VECTOR_TO_DIRECTION = {
    (0, -1): Direction.NORTH,
    (1, -1): Direction.NORTHEAST,
    (1, 0): Direction.EAST,
    (1, 1): Direction.SOUTHEAST,
    (0, 1): Direction.SOUTH,
    (-1, 1): Direction.SOUTHWEST,
    (-1, 0): Direction.WEST,
    (-1, -1): Direction.NORTHWEST,
}


class NonCooperativeStrategy(AntStrategy):
    """
    Omniscient cheating strategy.

    Features:
    - Full access to Environment
    - Exact ant position tracking
    - Global food/colony knowledge
    - Optimal A* pathfinding
    - Dynamic target selection
    - Pheromone deposition on optimal routes
    """

    def __init__(self):
        self.environment = None

        # Cached paths per ant
        self.current_paths = {}

        # Last target assigned to ant
        self.targets = {}

        self.path_cache = {}
        self.path_version = 0
        self.ant_last_pos = {}

    def set_environment(self, environment):
        """
        Receive direct access to full environment.
        """
        self.environment = environment

    def decide_action(self, perception: AntPerception) -> AntAction:
        ant = self._get_ant(perception.ant_id)

        if ant is None:
            return AntAction.NO_ACTION

        current_pos = (int(ant.x), int(ant.y))

        # -----------------------------
        # FOOD / COLONY ACTIONS
        # -----------------------------
        terrain = self.environment.get_terrain(*current_pos)

        if not ant.has_food and terrain == TerrainType.FOOD:
            return AntAction.PICK_UP_FOOD

        if ant.has_food and terrain == TerrainType.COLONY:
            return AntAction.DROP_FOOD

        # -----------------------------
        # PHEROMONE DEPOSITION
        # -----------------------------
        if ant.has_food:
            # Returning to colony
            if random.random() < 0.35:
                return AntAction.DEPOSIT_FOOD_PHEROMONE
        else:
            # Going to food
            if random.random() < 0.20:
                return AntAction.DEPOSIT_HOME_PHEROMONE

        # -----------------------------
        # TARGET SELECTION
        # -----------------------------
        if ant.has_food:
            target = self._nearest_colony(current_pos)
        else:
            target = self._best_food_target(current_pos)

        if target is None:
            return random.choice([
                AntAction.MOVE_FORWARD,
                AntAction.TURN_LEFT,
                AntAction.TURN_RIGHT,
            ])

        self.targets[perception.ant_id] = target

        # -----------------------------
        # PATH COMPUTATION
        # -----------------------------
        cache_key = (current_pos, target)

        path = self.path_cache.get(cache_key)

        if path is None:
            path = self._astar(current_pos, target)
            self.path_cache[cache_key] = path

        if not path or len(path) < 2:
            return AntAction.NO_ACTION

        self.current_paths[perception.ant_id] = path

        next_pos = path[1]

        desired_direction = self._direction_between(current_pos, next_pos)

        return self._move_towards(perception.direction, desired_direction)

    # =========================================================
    # PATHFINDING
    # =========================================================

    def _astar(self, start, goal):
        """
        Optimal A* pathfinding over full map.
        Supports diagonal movement.
        """

        env = self.environment

        frontier = []
        heapq.heappush(frontier, (0, start))

        came_from = {}
        cost_so_far = {}

        came_from[start] = None
        cost_so_far[start] = 0

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for neighbor in self._neighbors(current):
                new_cost = cost_so_far[current] + self._movement_cost(
                    current,
                    neighbor
                )

                if (
                    neighbor not in cost_so_far
                    or new_cost < cost_so_far[neighbor]
                ):
                    cost_so_far[neighbor] = new_cost

                    priority = (
                        new_cost
                        + self._heuristic(neighbor, goal)
                    )

                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return None

        # Reconstruct path
        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()

        return path

    def _neighbors(self, pos):
        x, y = pos

        result = []

        for direction in DIRECTION_ORDER:
            dx, dy = Direction.get_delta(direction)

            nx = x + dx
            ny = y + dy

            if self.environment.is_walkable(nx, ny):
                result.append((nx, ny))

        return result

    def _movement_cost(self, a, b):
        """
        Lower cost for cells already containing useful pheromones.
        """

        ax, ay = a
        bx, by = b

        diagonal = (ax != bx and ay != by)

        base_cost = 1.4 if diagonal else 1.0

        pheromone_bonus = (
            self.environment.food_pheromones.get_value(bx, by)
            + self.environment.home_pheromones.get_value(bx, by)
        ) * 0.002

        return max(0.1, base_cost - pheromone_bonus)

    def _heuristic(self, a, b):
        """
        Euclidean heuristic.
        """
        return math.hypot(b[0] - a[0], b[1] - a[1])

    # =========================================================
    # TARGET SELECTION
    # =========================================================

    def _best_food_target(self, current_pos):
        """
        Choose nearest remaining food source.
        """

        if not self.environment.food_positions:
            return None

        best_food = None
        best_score = float("inf")

        for food in self.environment.food_positions:
            score = self._heuristic(current_pos, food)

            if score < best_score:
                best_score = score
                best_food = food

        return best_food

    def _nearest_colony(self, current_pos):
        colonies = self.environment.colony_positions

        if not colonies:
            return None

        return min(
            colonies,
            key=lambda c: self._heuristic(current_pos, c)
        )

    # =========================================================
    # MOVEMENT
    # =========================================================

    def _direction_between(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        dx = 0 if dx == 0 else (1 if dx > 0 else -1)
        dy = 0 if dy == 0 else (1 if dy > 0 else -1)

        return VECTOR_TO_DIRECTION[(dx, dy)]

    def _move_towards(self, current_direction, target_direction):
        """
        Convert target direction into game action.
        """

        clockwise = (
            target_direction.value - current_direction.value
        ) % 8

        if clockwise == 0:
            return AntAction.MOVE_FORWARD

        if clockwise <= 4:
            return AntAction.TURN_RIGHT

        return AntAction.TURN_LEFT

    # =========================================================
    # ENVIRONMENT HELPERS
    # =========================================================

    def _get_ant(self, ant_id):
        for ant in self.environment.ants:
            if ant.id == ant_id:
                return ant
        return None