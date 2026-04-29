import math
import random
import heapq

from ant import AntAction, AntStrategy
from common import Direction, TerrainType
from environment import AntPerception


MAX_PATH_LENGTH = 600
MAX_KNOWN_CELLS = 1200
MAX_VISITED_CELLS = 700
MAX_MAP_SEARCH_NODES = 45
GATEWAY_SCAN_RADIUS = 1
GATEWAY_UPDATE_PERIOD = 5
MAX_GATEWAYS = 10
RECENT_WINDOW = 28
STAGNATION_RADIUS = 4
STAGNATION_UNIQUE_LIMIT = 10
FRONTIER_LOOKAHEAD = 3
TRAIL_LENGTH = 16
MAX_TRAIL_CELLS = 140
CARRIER_TRAIL_SCORE_WEIGHT = 2.8
CARRIER_TRAIL_EXPLORATION_WEIGHT = 1.4
CARRIER_TRAIL_MIN_ACTION_SCORE = 0.4


def current_terrain(perception):
    """Return the terrain under the ant."""
    return perception.visible_cells.get((0, 0))


def delta(direction):
    """Return the grid delta for a Direction."""
    return Direction.get_delta(direction)


def is_blocked(perception, direction=None):
    """Tell whether moving one step in a direction hits a wall or map border."""
    direction = direction or perception.direction
    dx, dy = delta(direction)
    terrain = perception.visible_cells.get((dx, dy))
    return terrain is None or terrain == TerrainType.WALL


def direction_from_delta(dx, dy):
    """Convert an arbitrary vector into one of the eight ant directions."""
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


def direction_to_closest(perception, terrain_type):
    """Find the direction of the closest visible terrain of a given type."""
    best = None
    best_distance = float("inf")
    for (dx, dy), terrain in perception.visible_cells.items():
        if terrain != terrain_type or (dx, dy) == (0, 0):
            continue
        distance = math.hypot(dx, dy)
        if distance < best_distance:
            best = direction_from_delta(dx, dy)
            best_distance = distance
    return best


def angular_distance(direction, target):
    """Return turn distance between two directions on the eight-direction ring."""
    diff = abs(direction.value - target.value) % 8
    return min(diff, 8 - diff)


def turn_towards(perception, target_direction):
    """Turn toward a target direction, or move if already aligned."""
    if target_direction is None:
        return random_turn()

    current = perception.direction.value
    target = target_direction.value
    clockwise = (target - current) % 8

    if clockwise == 0:
        return AntAction.MOVE_FORWARD if not is_blocked(perception) else random_turn()
    if clockwise == 4:
        return random_turn()
    return AntAction.TURN_RIGHT if clockwise <= 4 else AntAction.TURN_LEFT


def random_turn():
    """Pick a random left/right turn."""
    return random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])


def visible_free_distance(perception, direction, max_steps=8):
    """Count visible non-wall cells in a straight line."""
    dx, dy = delta(direction)
    distance = 0
    for step in range(1, max_steps + 1):
        terrain = perception.visible_cells.get((dx * step, dy * step))
        if terrain is None or terrain == TerrainType.WALL:
            break
        distance += 1
    return distance


def open_directions(perception):
    """List immediate directions that are not visibly blocked."""
    return [direction for direction in Direction if not is_blocked(perception, direction)]


def closest_food_carrier_offset(perception):
    """Return the closest visible ant carrying food, relative to this ant."""
    carriers = [offset for offset, has_food in perception.nearby_ants if has_food]
    if not carriers:
        return None
    return min(carriers, key=lambda offset: math.hypot(*offset))


def can_see_wall(perception):
    """Tell whether any wall is currently visible."""
    return TerrainType.WALL in perception.visible_cells.values()


class NonCooperativeStrategy(AntStrategy):
    """Independent strategy: personal memory, no pheromones."""

    def __init__(self):
        self.positions = {}
        self.outbound_paths = {}
        self.return_paths = {}
        self.food_paths = {}
        self.food_path_indices = {}
        self.last_actions = {}

        self.food_memory = {}
        self.pending_food_checks = {}
        self.carrier_trails = {}
        self.previous_unladen_neighbors = {}

        self.seen_ants = set()
        self.initial_heading = set()
        self.initial_direction_counts = {}
        self.sidestep_plans = {}

        self.recent_positions = {}
        self.known_maps = {}
        self.gateways = {}
        self.visit_counts = {}
        self.exhausted_zones = {}
        self.avoid_zones = {}
        self.farthest_points = {}

    def decide_action(self, perception: AntPerception) -> AntAction:
        """Update memory, handle pickup/drop, then choose a movement action."""
        self._update_position(perception)
        self._ensure_ant_state(perception)
        self._update_known_map(perception)
        self._remember_exploration_state(perception)
        self._update_carrier_transition_hint(perception)
        self._clear_empty_food_memory(perception)
        self._clear_reached_empty_food_target(perception)

        terrain = current_terrain(perception)

        if not perception.has_food and terrain == TerrainType.FOOD:
            self._remember_food(perception)
            return self._remember_action(perception, AntAction.PICK_UP_FOOD)

        if perception.has_food and terrain == TerrainType.COLONY:
            self._reset_after_drop(perception)
            return self._remember_action(perception, AntAction.DROP_FOOD)

        sidestep = self._initial_sidestep_action(perception)
        if sidestep is not None:
            return self._remember_action(perception, sidestep)

        action = self._move_with_food(perception) if perception.has_food else self._search_food(perception)
        return self._remember_action(perception, action)

    def _move_with_food(self, perception):
        """Return to the colony using visible colony, direct home, path memory, or map."""
        colony = direction_to_closest(perception, TerrainType.COLONY)
        if colony is not None:
            return turn_towards(perception, colony)

        home = self._home_direction(perception)
        if home is not None and not is_blocked(perception, home):
            return turn_towards(perception, home)

        reverse = self._return_path_direction(perception)
        if reverse is not None and not is_blocked(perception, reverse):
            return turn_towards(perception, reverse)

        gateway_home = self._gateway_direction_to(perception, (0, 0))
        if gateway_home is not None and not is_blocked(perception, gateway_home):
            return turn_towards(perception, gateway_home)

        mapped_home = self._mapped_home_direction(perception)
        if mapped_home is not None and not is_blocked(perception, mapped_home):
            return turn_towards(perception, mapped_home)

        target = reverse or mapped_home or home
        return self._best_open_action(perception, target=target, carrying_food=True)

    def _search_food(self, perception):
        """Search priority: visible food, initial heading, known path, carrier trail, map exploration."""
        visible_food = direction_to_closest(perception, TerrainType.FOOD)
        if visible_food is not None:
            self._end_initial_heading(perception)
            self._reset_food_path_progress(perception.ant_id)
            return turn_towards(perception, visible_food)

        if self._is_initial_heading(perception):
            if can_see_wall(perception) or is_blocked(perception):
                self._end_initial_heading(perception)
            else:
                return AntAction.MOVE_FORWARD

        remembered_food = self._remembered_food_action(perception)
        if remembered_food is not None:
            return remembered_food

        # High priority: observe and follow carrier trails (boosted importance)
        carrier_direction = self._observe_carrier_trail(perception)
        if carrier_direction is not None:
            self._end_initial_heading(perception)
            return self._trail_action(perception, carrier_direction)

        carrier_action = self._carrier_action(perception)
        if carrier_action is not None:
            return carrier_action

        if is_blocked(perception):
            return self._wall_escape_action(perception)

        if self._is_stagnating(perception):
            self._mark_current_area_to_avoid(perception)
            return self._start_unstuck(perception)

        return self._best_open_action(
            perception,
            target=self._sector_direction(perception),
            force_exploration=True,
        )

    def _observe_carrier_trail(self, perception):
        """Record an interesting unknown trail suggested by a visible food carrier."""
        if perception.can_see_food():
            return None

        carrier_offset = closest_food_carrier_offset(perception)
        if carrier_offset is None:
            return None

        direction = self._food_direction_from_carrier(perception, carrier_offset)
        self._remember_carrier_trail(perception, direction)
        return direction

    def _carrier_action(self, perception):
        """Follow the strongest useful carrier trail if one points to unknown cells."""
        direction = self._carrier_trail_direction(perception)
        if direction is None:
            return None

        self._end_initial_heading(perception)
        return self._trail_action(perception, direction)

    def _trail_action(self, perception, direction):
        """Turn into a carrier trail, detouring only when the next step is blocked."""
        if is_blocked(perception, direction):
            return self._best_open_action(perception, target=direction, force_exploration=False)
        return turn_towards(perception, direction)

    def _wall_escape_action(self, perception):
        """At each blockage, randomly choose between returning home and local exploration."""
        # Recompute this dilemma at every blockage: either try to come home, or
        # explore locally using the mapped frontier/gateway/trail scores.
        if random.random() < 0.5:
            target = self._home_direction(perception)
        else:
            target = None

        direction = self._mapped_random_direction(perception, target=target)
        if direction is not None:
            return turn_towards(perception, direction)

        return random_turn()

    def _best_open_action(self, perception, target=None, carrying_food=False, force_exploration=False):
        """Choose the best open direction and convert it to a turn/move action."""
        direction = self._best_open_direction(perception, target, carrying_food, force_exploration)
        return turn_towards(perception, direction)

    def _best_open_direction(self, perception, target=None, carrying_food=False, force_exploration=False):
        """Score open directions; use heavy map scores only during exploration."""
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        use_exploration_scores = not carrying_food and (force_exploration or target is None)

        def score(direction):
            dx, dy = delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            value = random.random() * 0.15
            value += visible_free_distance(perception, direction) * 0.08

            if direction == perception.direction:
                value += 0.35
            if target is not None:
                value -= angular_distance(direction, target) * 0.55
            if carrying_food:
                value -= math.hypot(*next_pos) * 0.08
            elif use_exploration_scores:
                value += self._exploration_score(ant_id, next_pos, force_exploration)
                value += self._map_frontier_score(ant_id, direction)
                value += self._gateway_score(ant_id, direction)
                value += self._carrier_trail_score(ant_id, direction) * CARRIER_TRAIL_EXPLORATION_WEIGHT
            return value

        return max(directions, key=score)

    def _exploration_score(self, ant_id, next_pos, force_exploration):
        """Score a candidate exploration cell using visits, avoid zones, and distance from old frontier."""
        value = 0.0
        if self._inside_avoid_zone(ant_id, next_pos):
            value -= 3.0 if force_exploration else 1.2
        value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 8) * 0.25

        farthest = self.farthest_points.get(ant_id)
        current = self.positions.get(ant_id, (0, 0))
        if farthest is not None:
            current_gap = math.hypot(current[0] - farthest[0], current[1] - farthest[1])
            next_gap = math.hypot(next_pos[0] - farthest[0], next_pos[1] - farthest[1])
            value += max(-1.0, min(1.0, next_gap - current_gap)) * 0.8

        if force_exploration:
            value += math.hypot(*next_pos) * 0.02
        return value

    def _update_position(self, perception):
        """Update the ant's odometry when the previous forward move was expected to succeed."""
        ant_id = perception.ant_id
        if ant_id is None:
            return

        self.positions.setdefault(ant_id, (0, 0))
        self.outbound_paths.setdefault(ant_id, [(0, 0)])

        last = self.last_actions.get(ant_id)
        if last is None:
            return

        action, direction, expected_move = last
        if action != AntAction.MOVE_FORWARD or not expected_move:
            return

        x, y = self.positions[ant_id]
        dx, dy = delta(direction)
        new_pos = (x + dx, y + dy)
        self.positions[ant_id] = new_pos

        if not perception.has_food:
            self._append_path_position(ant_id, new_pos)

    def _remember_action(self, perception, action):
        """Store the chosen action so the next tick can update memory consistently."""
        ant_id = perception.ant_id
        if ant_id is None:
            return action

        expected_move = action == AntAction.MOVE_FORWARD and not is_blocked(perception)
        self.last_actions[ant_id] = (action, perception.direction, expected_move)

        return action

    def _append_path_position(self, ant_id, position):
        """Append a new position to the outbound path with a bounded history."""
        path = self.outbound_paths.setdefault(ant_id, [(0, 0)])
        if not path or path[-1] != position:
            path.append(position)
        if len(path) > MAX_PATH_LENGTH:
            del path[: len(path) - MAX_PATH_LENGTH]

    def _remember_food(self, perception):
        """Store food location plus colony->food and food->colony paths."""
        ant_id = perception.ant_id
        if ant_id is None:
            return

        position = self.positions.get(ant_id, (0, 0))
        self.food_memory[ant_id] = position
        self._clear_exhausted_zone_near(ant_id, position)
        self.pending_food_checks[ant_id] = position
        self._append_path_position(ant_id, position)
        path_to_food = list(self.outbound_paths.get(ant_id, []))
        self.food_paths[ant_id] = path_to_food
        self.food_path_indices[ant_id] = 0
        self.return_paths[ant_id] = list(reversed(path_to_food))

    def _clear_empty_food_memory(self, perception):
        """Consume the one-step pickup check without deleting useful food memory."""
        ant_id = perception.ant_id
        if ant_id is None or ant_id not in self.pending_food_checks:
            return

        remembered = self.pending_food_checks.pop(ant_id)
        if perception.has_food and not perception.can_see_food() and self.food_memory.get(ant_id) == remembered:
            return

    def _clear_reached_empty_food_target(self, perception):
        """Forget a remembered food target if the ant returns there and finds it empty."""
        ant_id = perception.ant_id
        if ant_id is None or perception.has_food or perception.can_see_food():
            return

        target = self.food_memory.get(ant_id)
        if target is None:
            return

        current = self.positions.get(ant_id, (0, 0))
        if math.hypot(target[0] - current[0], target[1] - current[1]) > 3:
            return
        if current_terrain(perception) == TerrainType.FOOD:
            return

        self.food_memory.pop(ant_id, None)
        self.food_paths.pop(ant_id, None)
        self.food_path_indices.pop(ant_id, None)
        self._remember_exhausted_zone(ant_id, target, radius=8)

    def _reset_after_drop(self, perception):
        """Reset home odometry after a successful food drop."""
        ant_id = perception.ant_id
        if ant_id is None:
            return

        self.positions[ant_id] = (0, 0)
        self.outbound_paths[ant_id] = [(0, 0)]
        self.return_paths.pop(ant_id, None)
        self.recent_positions[ant_id] = [(0, 0)]
        self._reset_food_path_progress(ant_id)

    def _return_path_direction(self, perception):
        """Follow the reverse path from food back toward the colony."""
        ant_id = perception.ant_id
        path = self.return_paths.get(ant_id)
        if not path:
            return None

        current = self.positions.get(ant_id, (0, 0))
        nearest_index = min(
            range(len(path)),
            key=lambda index: math.hypot(path[index][0] - current[0], path[index][1] - current[1]),
        )
        nearest = path[nearest_index]
        nearest_distance = math.hypot(nearest[0] - current[0], nearest[1] - current[1])

        if nearest_distance <= 1.5:
            del path[: nearest_index + 1]
        else:
            del path[:nearest_index]

        if not path:
            return None

        target = path[0]
        return direction_from_delta(target[0] - current[0], target[1] - current[1])

    def _food_path_direction(self, perception):
        """Strictly follow the stored colony->food path using a per-ant progress index."""
        ant_id = perception.ant_id
        path = self.food_paths.get(ant_id)
        if not path:
            return None

        current = self.positions.get(ant_id, (0, 0))
        index = self.food_path_indices.get(ant_id, 0)
        while index < len(path) - 1 and path[index] == current:
            index += 1
        self.food_path_indices[ant_id] = index

        target = path[index]
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        if dx == 0 and dy == 0:
            return None
        return direction_from_delta(dx, dy)

    def _reset_food_path_progress(self, ant_id):
        """Restart progress along the stored colony->food path."""
        if ant_id in self.food_paths:
            self.food_path_indices[ant_id] = 0

    def _remembered_food_action(self, perception):
        """Prefer known food paths before any non-food exploration logic."""
        path = self.food_paths.get(perception.ant_id)
        if path:
            direction = self._food_path_direction(perception)
            if direction is None:
                return None

            self._end_initial_heading(perception)
            if is_blocked(perception, direction):
                return self._best_open_action(perception, target=direction, force_exploration=False)
            return turn_towards(perception, direction)

        for direction in (
            self._gateway_food_direction(perception),
            self._mapped_food_direction(perception),
            self._remembered_food_direction(perception),
        ):
            if direction is None:
                continue

            self._end_initial_heading(perception)
            if is_blocked(perception, direction):
                return self._best_open_action(perception, target=direction, force_exploration=False)
            return turn_towards(perception, direction)

        return None

    def _remembered_food_direction(self, perception):
        """Fallback vector toward remembered food when no path exists."""
        ant_id = perception.ant_id
        food = self.food_memory.get(ant_id)
        if food is None:
            return None
        if self._inside_exhausted_zone(ant_id, food):
            self.food_memory.pop(ant_id, None)
            self.food_paths.pop(ant_id, None)
            self.food_path_indices.pop(ant_id, None)
            return None

        current = self.positions.get(ant_id, (0, 0))
        dx = food[0] - current[0]
        dy = food[1] - current[1]
        if dx == 0 and dy == 0:
            return None
        return direction_from_delta(dx, dy)

    def _mapped_food_direction(self, perception):
        """Occasionally use the known map to route around walls toward remembered food."""
        food = self.food_memory.get(perception.ant_id)
        if food is None:
            return None
        if self._inside_exhausted_zone(perception.ant_id, food):
            return None
        if perception.steps_taken % 16 != 0:
            return None
        direct = self._remembered_food_direction(perception)
        if direct is not None and not is_blocked(perception, direct):
            return None
        return self._mapped_direction_to(perception.ant_id, food)

    def _gateway_food_direction(self, perception):
        """Use a remembered gateway as waypoint toward remembered food."""
        food = self.food_memory.get(perception.ant_id)
        if food is None or self._inside_exhausted_zone(perception.ant_id, food):
            return None
        return self._gateway_direction_to(perception, food)

    def _home_direction(self, perception):
        """Return the direct vector toward the colony origin in odometry space."""
        x, y = self.positions.get(perception.ant_id, (0, 0))
        if x == 0 and y == 0:
            return None
        return direction_from_delta(-x, -y)

    def _mapped_home_direction(self, perception):
        """Occasionally use the known map to route home around obstacles."""
        if perception.steps_taken % 16 != 0:
            return None
        return self._mapped_direction_to(perception.ant_id, (0, 0))

    def _food_direction_from_carrier(self, perception, carrier_offset):
        """Estimate the food-side direction suggested by a carrier's position."""
        ant_x, ant_y = self.positions.get(perception.ant_id, (0, 0))
        carrier_x = ant_x + carrier_offset[0]
        carrier_y = ant_y + carrier_offset[1]
        if carrier_x == 0 and carrier_y == 0:
            return direction_from_delta(-carrier_offset[0], -carrier_offset[1])
        return direction_from_delta(carrier_x, carrier_y)

    def _update_carrier_transition_hint(self, perception):
        """Detect ants that changed from empty to carrier nearby and record their likely trail."""
        ant_id = perception.ant_id
        if ant_id is None or perception.has_food:
            return

        previous = self.previous_unladen_neighbors.get(ant_id, [])
        for offset, has_food in perception.nearby_ants:
            if has_food and self._near_previous_unladen(offset, previous):
                direction = self._food_direction_from_carrier(perception, offset)
                self._remember_carrier_trail(perception, direction)
                self._end_initial_heading(perception)
                break

        self.previous_unladen_neighbors[ant_id] = [
            offset for offset, has_food in perception.nearby_ants if not has_food
        ]

    def _near_previous_unladen(self, carrier_offset, previous_offsets):
        """Match a current carrier to a previous nearby non-carrying ant observation."""
        return any(
            math.hypot(carrier_offset[0] - old[0], carrier_offset[1] - old[1]) <= 2
            for old in previous_offsets
        )

    def _remember_exploration_state(self, perception):
        """Update visits, recent positions, and farthest point memory."""
        ant_id = perception.ant_id
        if ant_id is None:
            return

        position = self.positions.get(ant_id, (0, 0))
        self.visit_counts.setdefault(ant_id, {})[position] = (
            self.visit_counts.setdefault(ant_id, {}).get(position, 0) + 1
        )
        if len(self.visit_counts[ant_id]) > MAX_VISITED_CELLS:
            self._trim_map_around(self.visit_counts[ant_id], position, MAX_VISITED_CELLS)
        recent = self.recent_positions.setdefault(ant_id, [])
        recent.append(position)
        if len(recent) > RECENT_WINDOW:
            del recent[: len(recent) - RECENT_WINDOW]

        farthest = self.farthest_points.get(ant_id)
        if farthest is None or math.hypot(*position) > math.hypot(*farthest):
            self.farthest_points[ant_id] = position

    def _is_stagnating(self, perception):
        """Detect whether the ant has stayed inside a small area for too long."""
        recent = self.recent_positions.get(perception.ant_id, [])
        if len(recent) < RECENT_WINDOW:
            return False

        center_x = sum(x for x, _ in recent) / len(recent)
        center_y = sum(y for _, y in recent) / len(recent)
        spread = max(math.hypot(x - center_x, y - center_y) for x, y in recent)
        return spread <= STAGNATION_RADIUS and len(set(recent)) <= STAGNATION_UNIQUE_LIMIT

    def _mark_current_area_to_avoid(self, perception):
        """Mark the current stagnation area as less attractive for exploration."""
        ant_id = perception.ant_id
        recent = self.recent_positions.get(ant_id, [])
        if not recent:
            return

        center = (
            sum(x for x, _ in recent) / len(recent),
            sum(y for _, y in recent) / len(recent),
        )
        self._add_avoid_zone(ant_id, center, radius=7)

    def _add_avoid_zone(self, ant_id, center, radius):
        """Remember a local area to penalize during exploration."""
        zones = self.avoid_zones.setdefault(ant_id, [])
        zone = (int(round(center[0])), int(round(center[1])), radius)
        if zone not in zones:
            zones.append(zone)
        if len(zones) > 8:
            del zones[: len(zones) - 8]

    def _remember_exhausted_zone(self, ant_id, center, radius):
        """Remember a food area that appears depleted."""
        zones = self.exhausted_zones.setdefault(ant_id, [])
        zone = (int(round(center[0])), int(round(center[1])), radius)
        if zone not in zones:
            zones.append(zone)
        self._clear_carrier_trails_near(ant_id, center, radius)
        self._add_avoid_zone(ant_id, center, radius)
        if len(zones) > 8:
            del zones[: len(zones) - 8]

    def _clear_exhausted_zone_near(self, ant_id, position):
        """Remove an exhausted marker if food is found there again."""
        zones = self.exhausted_zones.get(ant_id)
        if not zones:
            return
        self.exhausted_zones[ant_id] = [
            zone for zone in zones if math.hypot(position[0] - zone[0], position[1] - zone[1]) > zone[2]
        ]

    def _inside_exhausted_zone(self, ant_id, position):
        """Check whether a position lies inside a depleted food area."""
        for x, y, radius in self.exhausted_zones.get(ant_id, []):
            if math.hypot(position[0] - x, position[1] - y) <= radius:
                return True
        return False

    def _inside_avoid_zone(self, ant_id, position):
        """Check whether a position lies inside any exploration avoid zone."""
        for x, y, radius in self.avoid_zones.get(ant_id, []):
            if math.hypot(position[0] - x, position[1] - y) <= radius:
                return True
        return False

    def _update_known_map(self, perception):
        """Merge visible cells into the ant's personal map and refresh gateways."""
        ant_id = perception.ant_id
        if ant_id is None:
            return

        ant_x, ant_y = self.positions.get(ant_id, (0, 0))
        known_map = self.known_maps.setdefault(ant_id, {})
        for (dx, dy), terrain in perception.visible_cells.items():
            known_map[(ant_x + dx, ant_y + dy)] = terrain
        if len(known_map) > MAX_KNOWN_CELLS:
            self._trim_map_around(known_map, (ant_x, ant_y), MAX_KNOWN_CELLS)
        if can_see_wall(perception) and (
            perception.steps_taken % GATEWAY_UPDATE_PERIOD == 0
            or not self.gateways.get(ant_id)
        ):
            self._update_gateways(ant_id, (ant_x, ant_y), perception.visible_cells)

    def _update_gateways(self, ant_id, current, visible_cells):
        """Detect free cells near walls that likely act as useful passages."""
        known_map = self.known_maps.get(ant_id, {})
        gateways = self.gateways.setdefault(ant_id, set())
        cur_x, cur_y = current

        for dx, dy in visible_cells:
            position = (cur_x + dx, cur_y + dy)
            terrain = known_map.get(position)
            if terrain is None or terrain == TerrainType.WALL:
                continue
            if self._gateway_strength(known_map, position) >= 3 and not self._is_dead_end(known_map, position):
                gateways.add(position)
            elif position in gateways and self._is_dead_end(known_map, position):
                gateways.discard(position)

        if len(gateways) > MAX_GATEWAYS:
            closest = sorted(
                gateways,
                key=lambda position: math.hypot(position[0] - cur_x, position[1] - cur_y),
            )[:MAX_GATEWAYS]
            self.gateways[ant_id] = set(closest)

    def _gateway_strength(self, known_map, position):
        """Score how strongly a free cell looks like a passage through walls."""
        x, y = position
        wall_count = 0
        free_count = 0
        for dx in range(-GATEWAY_SCAN_RADIUS, GATEWAY_SCAN_RADIUS + 1):
            for dy in range(-GATEWAY_SCAN_RADIUS, GATEWAY_SCAN_RADIUS + 1):
                if dx == 0 and dy == 0:
                    continue
                terrain = known_map.get((x + dx, y + dy))
                if terrain == TerrainType.WALL:
                    wall_count += 1
                elif terrain is not None:
                    free_count += 1

        has_vertical_wall = (
            known_map.get((x - 1, y)) == TerrainType.WALL
            or known_map.get((x + 1, y)) == TerrainType.WALL
        )
        has_horizontal_wall = (
            known_map.get((x, y - 1)) == TerrainType.WALL
            or known_map.get((x, y + 1)) == TerrainType.WALL
        )
        return wall_count if free_count >= 2 and (has_vertical_wall or has_horizontal_wall) else 0

    def _is_dead_end(self, known_map, start):
        """Reject gateway candidates that look like local cul-de-sacs."""
        open_directions_count = 0
        unknown_directions_count = 0
        wall_directions_count = 0

        for direction in Direction:
            dx, dy = delta(direction)
            terrain = known_map.get((start[0] + dx, start[1] + dy))
            if terrain is None:
                unknown_directions_count += 1
            elif terrain == TerrainType.WALL:
                wall_directions_count += 1
            else:
                open_directions_count += 1

        if open_directions_count <= 1:
            return True
        return open_directions_count <= 2 and unknown_directions_count == 0 and wall_directions_count >= 5

    def _mapped_direction_to(self, ant_id, target):
        """Bounded A* over known non-wall cells toward a mapped target."""
        if ant_id is None:
            return None

        known_map = self.known_maps.get(ant_id, {})
        visits = self.visit_counts.get(ant_id, {})
        current = self.positions.get(ant_id, (0, 0))
        if current == target:
            return None
        if target not in known_map:
            return None

        queue = [(0.0, 0.0, current)]
        came_from = {current: None}
        costs = {current: 0.0}

        searched = 0
        while queue and searched < MAX_MAP_SEARCH_NODES:
            _, current_cost, position = heapq.heappop(queue)
            if current_cost > costs.get(position, float("inf")):
                continue
            searched += 1
            if position == target:
                break

            for direction in Direction:
                dx, dy = delta(direction)
                neighbor = (position[0] + dx, position[1] + dy)
                if known_map.get(neighbor) == TerrainType.WALL:
                    continue
                if neighbor not in known_map and neighbor != target:
                    continue

                step_cost = 1.0
                step_cost += min(visits.get(neighbor, 0), 8) * 0.12
                if neighbor in self.gateways.get(ant_id, set()):
                    step_cost -= 0.25

                new_cost = current_cost + max(0.2, step_cost)
                if new_cost >= costs.get(neighbor, float("inf")):
                    continue

                costs[neighbor] = new_cost
                came_from[neighbor] = position
                heuristic = math.hypot(target[0] - neighbor[0], target[1] - neighbor[1])
                heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))

        if target not in came_from:
            return None

        step = target
        while came_from[step] is not None and came_from[step] != current:
            step = came_from[step]

        return direction_from_delta(step[0] - current[0], step[1] - current[1])

    def _gateway_direction_to(self, perception, target):
        """Aim at a useful gateway between the ant and a far target."""
        ant_id = perception.ant_id
        if ant_id is None:
            return None

        current = self.positions.get(ant_id, (0, 0))
        gateway = self._best_gateway_between(ant_id, current, target)
        if gateway is None:
            return None
        if math.hypot(gateway[0] - current[0], gateway[1] - current[1]) <= 2:
            return direction_from_delta(target[0] - current[0], target[1] - current[1])
        return direction_from_delta(gateway[0] - current[0], gateway[1] - current[1])

    def _best_gateway_between(self, ant_id, current, target):
        """Pick the most promising gateway roughly between current and target."""
        gateways = self.gateways.get(ant_id, set())
        if not gateways:
            return None

        vector_x = target[0] - current[0]
        vector_y = target[1] - current[1]
        target_distance = math.hypot(vector_x, vector_y)
        if target_distance < 8:
            return None

        candidates = []
        for gateway in gateways:
            current_to_gateway = math.hypot(gateway[0] - current[0], gateway[1] - current[1])
            gateway_to_target = math.hypot(target[0] - gateway[0], target[1] - gateway[1])
            if current_to_gateway < 3:
                continue
            if current_to_gateway + gateway_to_target > target_distance * 1.8 + 12:
                continue

            dot = (gateway[0] - current[0]) * vector_x + (gateway[1] - current[1]) * vector_y
            if dot <= 0:
                continue

            visits = self.visit_counts.get(ant_id, {}).get(gateway, 0)
            score = current_to_gateway + gateway_to_target + visits * 1.5
            candidates.append((score, gateway))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _map_frontier_score(self, ant_id, direction):
        """Reward directions that reveal unknown cells and avoid mapped walls/revisits."""
        if ant_id is None:
            return 0.0

        known_map = self.known_maps.get(ant_id, {})
        visits = self.visit_counts.get(ant_id, {})
        current = self.positions.get(ant_id, (0, 0))
        dx, dy = delta(direction)

        score = 0.0
        for step in range(1, FRONTIER_LOOKAHEAD + 1):
            position = (current[0] + dx * step, current[1] + dy * step)
            terrain = known_map.get(position)
            if terrain == TerrainType.WALL:
                score -= 1.5 / step
                break
            if terrain is None:
                score += 1.4 / step
            else:
                score -= min(visits.get(position, 0), 6) * 0.18 / step

            for side in (-1, 1):
                side_direction = Direction((direction.value + side) % 8)
                sx, sy = delta(side_direction)
                side_position = (position[0] + sx, position[1] + sy)
                if side_position not in known_map:
                    score += 0.35 / step

        return score

    def _gateway_score(self, ant_id, direction):
        """Reward directions aligned with nearby remembered gateways."""
        gateways = self.gateways.get(ant_id, set())
        if not gateways:
            return 0.0

        current = self.positions.get(ant_id, (0, 0))
        best = 0.0
        for gateway in gateways:
            distance = math.hypot(gateway[0] - current[0], gateway[1] - current[1])
            if distance > 18:
                continue
            gateway_direction = direction_from_delta(gateway[0] - current[0], gateway[1] - current[1])
            alignment = 1.0 / (1 + angular_distance(direction, gateway_direction))
            best = max(best, alignment * (1.5 / (1 + distance * 0.15)))
        return best

    def _remember_carrier_trail(self, perception, direction):
        """Store an unknown trail suggested by a carrier coming from food."""
        ant_id = perception.ant_id
        if ant_id is None or direction is None:
            return

        current = self.positions.get(ant_id, (0, 0))
        known_map = self.known_maps.get(ant_id, {})
        trail = self.carrier_trails.setdefault(ant_id, {})
        dx, dy = delta(direction)

        for step in range(1, TRAIL_LENGTH + 1):
            position = (current[0] + dx * step, current[1] + dy * step)
            if self._inside_exhausted_zone(ant_id, position):
                break
            if position in known_map:
                continue
            strength = 1.2 - step / (TRAIL_LENGTH + 5)
            trail[position] = max(trail.get(position, 0.0), strength)

        if len(trail) > MAX_TRAIL_CELLS:
            strongest = sorted(trail.items(), key=lambda item: item[1], reverse=True)[:MAX_TRAIL_CELLS]
            self.carrier_trails[ant_id] = dict(strongest)

    def _carrier_trail_score(self, ant_id, direction):
        """Reward directions that point into unknown zones highlighted by carriers."""
        trail = self.carrier_trails.get(ant_id)
        if not trail:
            return 0.0

        current = self.positions.get(ant_id, (0, 0))
        known_map = self.known_maps.get(ant_id, {})
        dx, dy = delta(direction)
        score = 0.0
        for step in range(1, 9):
            position = (current[0] + dx * step, current[1] + dy * step)
            if position in known_map:
                continue
            score += trail.get(position, 0.0) / step

        return score * CARRIER_TRAIL_SCORE_WEIGHT

    def _carrier_trail_direction(self, perception):
        """Choose the best immediate direction suggested by carrier trails."""
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        scored = [
            (self._carrier_trail_score(ant_id, direction), direction)
            for direction in directions
        ]
        best_score, best_direction = max(scored, key=lambda item: item[0])
        if best_score < CARRIER_TRAIL_MIN_ACTION_SCORE:
            return None
        return best_direction

    def _clear_carrier_trails_near(self, ant_id, center, radius):
        """Remove carrier trails that point into newly depleted food zones."""
        trail = self.carrier_trails.get(ant_id)
        if not trail:
            return

        self.carrier_trails[ant_id] = {
            position: value
            for position, value in trail.items()
            if math.hypot(position[0] - center[0], position[1] - center[1]) > radius
        }

    def _trim_map_around(self, mapping, center, max_size):
        """Keep only the nearest remembered entries when a memory table grows too large."""
        keep = sorted(
            mapping,
            key=lambda position: math.hypot(position[0] - center[0], position[1] - center[1]),
        )[:max_size]
        keep = set(keep)
        for position in list(mapping):
            if position not in keep:
                del mapping[position]

    def _mapped_random_direction(self, perception, target=None):
        """Randomly choose an open direction, weighted by map frontier/gateway/trail value."""
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        weights = []
        for direction in directions:
            weight = 1.0 + max(0.0, self._map_frontier_score(ant_id, direction))
            weight += max(0.0, self._gateway_score(ant_id, direction))
            weight += max(0.0, self._carrier_trail_score(ant_id, direction))
            if direction == perception.direction:
                weight += 0.8
            if target is not None:
                weight += 2.0 / (1 + angular_distance(direction, target))
            weights.append(weight)

        return random.choices(directions, weights=weights, k=1)[0]

    def _start_unstuck(self, perception):
        """Choose one exploration-biased step away from a stagnation zone."""
        direction = self._best_open_direction(perception, force_exploration=True)
        if direction is None:
            return random_turn()

        return turn_towards(perception, direction)

    def _sector_direction(self, perception):
        """Give each ant a stable coarse exploration sector."""
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
        dx, dy = sectors[(perception.ant_id or 0) % len(sectors)]
        return direction_from_delta(dx, dy)

    def _ensure_ant_state(self, perception):
        """Initialize per-ant first-heading and sidestep state."""
        ant_id = perception.ant_id
        if ant_id is None or ant_id in self.seen_ants:
            return

        self.seen_ants.add(ant_id)
        self.initial_heading.add(ant_id)
        self._plan_initial_sidestep(perception)

    def _is_initial_heading(self, perception):
        """Tell whether this ant is still in its initial straight-line phase."""
        return perception.ant_id in self.initial_heading

    def _end_initial_heading(self, perception):
        """End the initial straight-line phase for an ant."""
        self.initial_heading.discard(perception.ant_id)

    def _plan_initial_sidestep(self, perception):
        """Assign small alternating sidesteps to ants born with duplicate directions."""
        ant_id = perception.ant_id
        direction_value = perception.direction.value
        count = self.initial_direction_counts.get(direction_value, 0)
        self.initial_direction_counts[direction_value] = count + 1

        if count == 0:
            return

        if count % 2:
            self.sidestep_plans[ant_id] = [AntAction.TURN_LEFT, AntAction.MOVE_FORWARD, AntAction.TURN_RIGHT]
        else:
            self.sidestep_plans[ant_id] = [AntAction.TURN_RIGHT, AntAction.MOVE_FORWARD, AntAction.TURN_LEFT]

    def _initial_sidestep_action(self, perception):
        """Execute one step of the initial sidestep plan, if any."""
        ant_id = perception.ant_id
        plan = self.sidestep_plans.get(ant_id)
        if not plan:
            return None

        action = plan.pop(0)

        if action == AntAction.MOVE_FORWARD and is_blocked(perception):
            action = plan.pop(0) if plan else None

        if not plan:
            self.sidestep_plans.pop(ant_id, None)
        return action
