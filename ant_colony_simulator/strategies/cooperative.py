import math
import random

from ant import AntAction, AntStrategy
from common import Direction, TerrainType
from environment import AntPerception


FOOD_PHEROMONE_THRESHOLD = 10.0
HOME_PHEROMONE_THRESHOLD = 25.0
FOOD_DEPOSIT_INTERVAL = 3
HOME_DEPOSIT_INTERVAL = 10
PHEROMONE_FOLLOW_PROBABILITY = 0.75
FOOD_SEEN_DEPOSIT_INTERVAL = 2


def current_terrain(perception):
    """Return the terrain under the ant."""
    return perception.visible_cells.get((0, 0))


def relative_for_direction(direction):
    """Return the one-cell delta for a direction."""
    return Direction.get_delta(direction)


def is_blocked(perception, direction=None):
    """Treat walls and unseen grid borders as blocked cells."""
    direction = direction or perception.direction
    dx, dy = relative_for_direction(direction)
    terrain = perception.visible_cells.get((dx, dy))
    return terrain is None or terrain == TerrainType.WALL


def turn_towards(perception, target_direction):
    """Turn one step toward a direction, or move if already aligned."""
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
    """Find the closest visible terrain of a given type."""
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


def distance_to_closest(perception, terrain_type):
    """Return the distance to the closest visible terrain of a given type."""
    best_distance = float("inf")
    for (dx, dy), cell in perception.visible_cells.items():
        if cell == terrain_type and (dx, dy) != (0, 0):
            best_distance = min(best_distance, math.hypot(dx, dy))
    return best_distance


def angular_distance(direction, target):
    """Return the number of turns between two directions."""
    diff = abs(direction.value - target.value) % 8
    return min(diff, 8 - diff)


def opposite_direction(direction):
    """Return the direction opposite to another direction."""
    return Direction((direction.value + 4) % 8)


def open_directions(perception):
    """List all immediately walkable directions."""
    return [direction for direction in Direction if not is_blocked(perception, direction)]


def pheromone_direction(perception, pheromone_map, threshold, away=False):
    """Choose a direction probabilistically from the local pheromone gradient."""
    candidates = []

    for direction in open_directions(perception):
        score = pheromone_score(perception, pheromone_map, direction, away=away)
        if direction == perception.direction:
            score *= 1.12
        if score >= threshold:
            candidates.append((score, direction))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score = candidates[0][0]
    useful = [item for item in candidates if item[0] >= best_score * 0.65]
    weights = [score for score, _ in useful]
    return random.choices([direction for _, direction in useful], weights=weights, k=1)[0]


def pheromone_score(perception, pheromone_map, candidate_direction, away=False):
    """Score how well a candidate direction follows a visible pheromone cone."""
    score = 0.0

    for (dx, dy), value in pheromone_map.items():
        if value <= 0 or (dx, dy) == (0, 0):
            continue
        if perception.visible_cells.get((dx, dy)) == TerrainType.WALL:
            continue

        cell_direction = direction_from_delta(dx, dy)
        target_direction = Direction((cell_direction.value + 4) % 8) if away else cell_direction
        angle = angular_distance(candidate_direction, target_direction)
        if angle > 2:
            continue

        distance = max(1.0, math.hypot(dx, dy))
        alignment = (3 - angle) / 3
        score += value * alignment / distance

    return score


def random_safe_move(perception, forward_bias=0.55):
    """Move forward when safe, otherwise turn randomly."""
    if not is_blocked(perception) and random.random() < forward_bias:
        return AntAction.MOVE_FORWARD
    return random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])


def open_direction(perception, preferred_direction=None):
    """Pick a walkable direction, biased by preference and forward movement."""
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


def exploration_direction(perception, preferred_direction=None):
    """Explore without map memory, using openness, sector bias, and small randomness."""
    candidates = []

    for direction in Direction:
        dx, dy = relative_for_direction(direction)
        terrain = perception.visible_cells.get((dx, dy))
        if terrain is None or terrain == TerrainType.WALL:
            continue

        score = random.random() * 0.35
        score += visible_free_distance(perception, direction) * 0.18
        if preferred_direction is not None:
            score -= angular_distance(direction, preferred_direction) * 0.18
        if direction == perception.direction:
            score += 0.45
        candidates.append((score, direction))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def visible_free_distance(perception, direction, max_steps=3):
    """Count visible free cells ahead of a direction."""
    dx, dy = relative_for_direction(direction)
    distance = 0
    for step in range(1, max_steps + 1):
        terrain = perception.visible_cells.get((dx * step, dy * step))
        if terrain is None or terrain == TerrainType.WALL:
            break
        distance += 1
    return distance


def direction_from_delta(dx, dy):
    """Convert a vector into one of the eight movement directions."""
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


class CooperativeStrategy(AntStrategy):
    """Cooperative strategy using pheromones, without spatial memory."""

    def __init__(self):
        pass

    def decide_action(self, perception: AntPerception) -> AntAction:
        """Follow local targets first, then deposit/sign-read pheromones."""
        terrain = current_terrain(perception)

        if not perception.has_food and terrain == TerrainType.FOOD:
            return AntAction.PICK_UP_FOOD

        if perception.has_food and terrain == TerrainType.COLONY:
            return AntAction.DROP_FOOD

        direct_action = self._direct_target_action(perception)
        if direct_action is not None:
            return direct_action

        if self._should_deposit(perception):
            if perception.has_food:
                return AntAction.DEPOSIT_FOOD_PHEROMONE
            return AntAction.DEPOSIT_HOME_PHEROMONE

        return self._decide_movement(perception)

    def _direct_target_action(self, perception: AntPerception) -> AntAction | None:
        """Move toward food/colony when it is directly visible."""
        if perception.has_food:
            colony_direction = direction_to_closest(perception, TerrainType.COLONY)
            if colony_direction is not None:
                return turn_towards(perception, colony_direction)

            food_direction = direction_to_closest(perception, TerrainType.FOOD)
            if food_direction is not None:
                return turn_towards(perception, opposite_direction(food_direction))

        else:
            food_direction = direction_to_closest(perception, TerrainType.FOOD)
            if food_direction is not None:
                if self._should_deposit_home_before_food(perception):
                    return AntAction.DEPOSIT_HOME_PHEROMONE
                return turn_towards(perception, food_direction)
        return None

    def _decide_movement(self, perception: AntPerception) -> AntAction:
        """Use the right pheromone gradient, then fallback to stateless exploration."""
        if perception.has_food:
            home_direction = self._probabilistic_pheromone_direction(
                perception,
                perception.home_pheromone,
                HOME_PHEROMONE_THRESHOLD,
                away=True,
            )
            if home_direction is not None:
                return self._move_or_escape(perception, home_direction)

            return self._move_or_escape(
                perception,
                exploration_direction(perception, preferred_direction=perception.direction),
            )
        else:
            food_trail_direction = self._probabilistic_pheromone_direction(
                perception,
                perception.food_pheromone,
                FOOD_PHEROMONE_THRESHOLD,
            )
            if food_trail_direction is not None:
                return self._move_or_escape(perception, food_trail_direction)

            sector_direction = self._sector_direction(perception, returning=False)
            if sector_direction is not None:
                return self._move_or_escape(perception, sector_direction)

        return turn_towards(perception, exploration_direction(perception))

    def _probabilistic_pheromone_direction(self, perception, pheromone_map, threshold, away=False):
        """Usually follow pheromones, sometimes diversify with local exploration."""
        direction = pheromone_direction(perception, pheromone_map, threshold, away=away)
        if direction is None:
            return None
        if random.random() < PHEROMONE_FOLLOW_PROBABILITY:
            return direction
        return exploration_direction(perception, preferred_direction=direction)

    def _should_deposit(self, perception: AntPerception) -> bool:
        """Deposit often enough to form gradients, staggered by ant id."""
        if perception.steps_taken <= 1:
            return False

        ant_offset = perception.ant_id or 0
        if perception.has_food:
            if perception.can_see_food():
                return (perception.steps_taken + ant_offset) % FOOD_SEEN_DEPOSIT_INTERVAL == 0
            return (perception.steps_taken + ant_offset) % FOOD_DEPOSIT_INTERVAL == 0

        if perception.can_see_colony():
            return False
        return (perception.steps_taken + ant_offset) % HOME_DEPOSIT_INTERVAL == 0

    def _should_deposit_home_before_food(self, perception: AntPerception) -> bool:
        """Mark the return trail only when very close to visible food."""
        if perception.steps_taken <= 1 or perception.can_see_colony():
            return False
        if distance_to_closest(perception, TerrainType.FOOD) > 1.5:
            return False
        ant_offset = perception.ant_id or 0
        return (perception.steps_taken + ant_offset) % 2 == 0

    def _move_or_escape(self, perception: AntPerception, direction) -> AntAction:
        """Move toward a direction, or escape locally if it is blocked."""
        if is_blocked(perception) or is_blocked(perception, direction):
            return self._wall_escape_action(perception, preferred_direction=direction)
        return turn_towards(perception, direction)

    def _wall_escape_action(self, perception: AntPerception, preferred_direction=None) -> AntAction:
        """Break wall oscillations without remembering positions."""
        direction = open_direction(perception, preferred_direction=preferred_direction)
        if direction is not None:
            return turn_towards(perception, direction)

        return random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])

    def _sector_direction(self, perception: AntPerception, returning=False):
        """Spread ants across fixed sectors using only their id."""
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
        if returning:
            dx, dy = -dx, -dy
        return direction_from_delta(dx, dy)
