from collections import defaultdict, deque
import math
import random

from ant import AntAction, AntStrategy
from common import Direction, TerrainType
from environment import AntPerception


def current_terrain(perception):
    return perception.visible_cells.get((0, 0))


def relative_for_direction(direction):
    return Direction.get_delta(direction)


def is_blocked(perception, direction=None):
    direction = direction or perception.direction
    dx, dy = relative_for_direction(direction)
    terrain = perception.visible_cells.get((dx, dy))
    # A missing adjacent cell is treated as a border wall by this strategy.
    return terrain is None or terrain == TerrainType.WALL


def turn_towards(perception, target_direction):
    if target_direction is None:
        return random_safe_move(perception)

    current = perception.direction.value
    target = target_direction.value if isinstance(target_direction, Direction) else target_direction
    clockwise = (target - current) % 8

    if clockwise == 0:
        if not is_blocked(perception):
            return AntAction.MOVE_FORWARD
        return random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])

    if clockwise <= 4:
        return AntAction.TURN_RIGHT
    return AntAction.TURN_LEFT


def direction_to_closest(perception, terrain_type):
    best_direction = None
    best_distance = float("inf")

    for (dx, dy), cell in perception.visible_cells.items():
        if cell != terrain_type or (dx, dy) == (0, 0):
            continue

        distance = math.hypot(dx, dy)
        if distance < best_distance:
            best_distance = distance
            best_direction = direction_from_delta(dx, dy)

    return best_direction


def strongest_pheromone_direction(perception, pheromone_map, avoid_terrain=None):
    best_direction = None
    best_score = 0.0
    avoid_terrain = avoid_terrain or set()

    for (dx, dy), value in pheromone_map.items():
        if value <= 0 or (dx, dy) == (0, 0):
            continue

        terrain = perception.visible_cells.get((dx, dy))
        if terrain in avoid_terrain:
            continue

        distance = max(1.0, math.hypot(dx, dy))
        score = value / distance
        if score > best_score:
            best_score = score
            best_direction = direction_from_delta(dx, dy)

    return best_direction


def random_safe_move(perception, forward_bias=0.55):
    if not is_blocked(perception) and random.random() < forward_bias:
        return AntAction.MOVE_FORWARD
    return random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])


def open_direction(perception, preferred_direction=None):
    candidates = []

    for direction in Direction:
        dx, dy = relative_for_direction(direction)
        terrain = perception.visible_cells.get((dx, dy))
        if terrain is None or terrain == TerrainType.WALL:
            continue

        score = random.random()
        if direction == preferred_direction:
            score += 1.0
        if direction == perception.direction:
            score += 0.3
        candidates.append((score, direction))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def exploration_direction(perception, memory=None):
    candidates = []
    visited = memory or set()

    for direction in Direction:
        dx, dy = relative_for_direction(direction)
        terrain = perception.visible_cells.get((dx, dy))
        if terrain is None or terrain == TerrainType.WALL:
            continue

        score = random.random()
        if (dx, dy) not in visited and direction.value not in visited:
            score += 0.8
        if direction == perception.direction:
            score += 0.4
        candidates.append((score, direction))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def direction_from_delta(dx, dy):
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)

    mapping = {
        (0, -1): Direction.NORTH,
        (1, -1): Direction.NORTHEAST,
        (1, 0): Direction.EAST,
        (1, 1): Direction.SOUTHEAST,
        (0, 1): Direction.SOUTH,
        (-1, 1): Direction.SOUTHWEST,
        (-1, 0): Direction.WEST,
        (-1, -1): Direction.NORTHWEST,
    }
    return mapping.get((step_x, step_y), Direction.NORTH)


class SmartStrategy(AntStrategy):
    """Balanced strategy using direct perception, pheromones, and loop avoidance."""

    def __init__(self):
        self.deposit_interval = 3
        self.recent_directions = defaultdict(lambda: deque(maxlen=4))
        self.positions = {}
        self.last_moves = {}
        self.wall_escape_turns = {}

    def decide_action(self, perception: AntPerception) -> AntAction:
        self._update_position(perception)
        terrain = current_terrain(perception)

        if not perception.has_food and terrain == TerrainType.FOOD:
            return self._remember_action(perception, AntAction.PICK_UP_FOOD)

        if perception.has_food and terrain == TerrainType.COLONY:
            self.positions[perception.ant_id] = (0, 0)
            return self._remember_action(perception, AntAction.DROP_FOOD)

        direct_action = self._direct_target_action(perception)
        if direct_action is not None:
            return self._remember_action(perception, direct_action)

        if self._should_deposit(perception):
            if perception.has_food:
                return self._remember_action(perception, AntAction.DEPOSIT_FOOD_PHEROMONE)
            return self._remember_action(perception, AntAction.DEPOSIT_HOME_PHEROMONE)

        return self._remember_action(perception, self._decide_movement(perception))

    def _direct_target_action(self, perception: AntPerception) -> AntAction | None:
        if perception.has_food:
            colony_direction = direction_to_closest(perception, TerrainType.COLONY)
            if colony_direction is not None:
                self._remember_direction(perception, colony_direction)
                return turn_towards(perception, colony_direction)
        else:
            food_direction = direction_to_closest(perception, TerrainType.FOOD)
            if food_direction is not None:
                self._remember_direction(perception, food_direction)
                return turn_towards(perception, food_direction)
        return None

    def _decide_movement(self, perception: AntPerception) -> AntAction:
        if perception.has_food:
            home_direction = strongest_pheromone_direction(
                perception,
                perception.home_pheromone,
                avoid_terrain={TerrainType.WALL},
            )
            if home_direction is not None:
                self._remember_direction(perception, home_direction)
                return self._move_or_escape(perception, home_direction)

            estimated_home = self._direction_home(perception)
            if estimated_home is not None:
                self._remember_direction(perception, estimated_home)
                return self._move_or_escape(perception, estimated_home)
        else:
            food_trail_direction = strongest_pheromone_direction(
                perception,
                perception.food_pheromone,
                avoid_terrain={TerrainType.WALL, TerrainType.COLONY},
            )
            if food_trail_direction is not None:
                self._remember_direction(perception, food_trail_direction)
                return self._move_or_escape(perception, food_trail_direction)

            sector_direction = self._sector_direction(perception)
            if sector_direction is not None:
                self._remember_direction(perception, sector_direction)
                return self._move_or_escape(perception, sector_direction)

        memory = self._opposite_directions(perception)
        direction = exploration_direction(perception, memory)
        self._remember_direction(perception, direction)
        return self._move_or_escape(perception, direction)

    def _should_deposit(self, perception: AntPerception) -> bool:
        if perception.steps_taken <= 1:
            return False
        ant_offset = perception.ant_id or 0
        return (perception.steps_taken + ant_offset) % self.deposit_interval == 0

    def _move_or_escape(self, perception: AntPerception, direction) -> AntAction:
        if direction is not None and (is_blocked(perception) or is_blocked(perception, direction)):
            return self._wall_escape_action(perception, preferred_direction=direction)
        return turn_towards(perception, direction)

    def _wall_escape_action(self, perception: AntPerception, preferred_direction=None) -> AntAction:
        ant_id = perception.ant_id
        if ant_id is not None and self.wall_escape_turns.get(ant_id):
            turns_left, action = self.wall_escape_turns[ant_id]
            if turns_left <= 1:
                del self.wall_escape_turns[ant_id]
            else:
                self.wall_escape_turns[ant_id] = (turns_left - 1, action)
            return action

        direction = open_direction(perception, preferred_direction=preferred_direction)
        if direction is not None:
            return turn_towards(perception, direction)

        action = random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])
        if ant_id is not None:
            self.wall_escape_turns[ant_id] = (random.randint(2, 4), action)
        return action

    def _remember_direction(self, perception: AntPerception, direction) -> None:
        ant_id = perception.ant_id
        if ant_id is None or direction is None:
            return
        self.recent_directions[ant_id].append(direction.value)

    def _opposite_directions(self, perception: AntPerception) -> set[int]:
        ant_id = perception.ant_id
        if ant_id is None:
            return set()
        return {(direction + 4) % 8 for direction in self.recent_directions[ant_id]}

    def _update_position(self, perception: AntPerception) -> None:
        ant_id = perception.ant_id
        if ant_id is None:
            return
        self.positions.setdefault(ant_id, (0, 0))
        last = self.last_moves.get(ant_id)
        if last is None:
            return

        action, direction = last
        if action == AntAction.MOVE_FORWARD:
            x, y = self.positions[ant_id]
            dx, dy = relative_for_direction(direction)
            self.positions[ant_id] = (x + dx, y + dy)

    def _remember_action(self, perception: AntPerception, action: AntAction) -> AntAction:
        if perception.ant_id is not None:
            self.last_moves[perception.ant_id] = (action, perception.direction)
        return action

    def _direction_home(self, perception: AntPerception):
        x, y = self.positions.get(perception.ant_id, (0, 0))
        if x == 0 and y == 0:
            return None
        return direction_from_delta(-x, -y)

    def _sector_direction(self, perception: AntPerception):
        ant_id = perception.ant_id or 1
        sectors = [
            (1, -1),
            (-1, -1),
            (1, 1),
            (-1, 1),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ]
        dx, dy = sectors[ant_id % len(sectors)]
        return direction_from_delta(dx, dy)
