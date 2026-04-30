import heapq
import math
import random

from ant import AntAction, AntStrategy
from common import Direction, TerrainType
from environment import AntPerception


MAX_PATH_LENGTH = 800
MAX_KNOWN_CELLS = 1800
MAX_VISITED_CELLS = 900
MAX_ASTAR_NODES = 90
MAX_GATEWAYS = 20
MAX_PHEROMONE_SPOTS = 30
RECENT_WINDOW = 26
STAGNATION_RADIUS = 4
STAGNATION_UNIQUE_LIMIT = 10
GATEWAY_SCAN_RADIUS = 7
GATEWAY_DETECTION_PERIOD = 3
FRONTIER_LOOKAHEAD = 4

ROLE_CARRIER = "carrier"
ROLE_EXPLORER = "explorer"


def current_terrain(perception):
    """Return the terrain under the ant."""
    return perception.visible_cells.get((0, 0))


def delta(direction):
    """Return the one-cell delta for a Direction."""
    return Direction.get_delta(direction)


def is_blocked(perception, direction=None):
    """Treat walls and unseen grid borders as blocked."""
    direction = direction or perception.direction
    dx, dy = delta(direction)
    terrain = perception.visible_cells.get((dx, dy))
    return terrain is None or terrain == TerrainType.WALL


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


def angular_distance(direction, target):
    """Return turn distance on the eight-direction ring."""
    diff = abs(direction.value - target.value) % 8
    return min(diff, 8 - diff)


def turn_towards(perception, target):
    """Turn toward a target direction, or move when already aligned."""
    if target is None:
        return random_turn()

    clockwise = (target.value - perception.direction.value) % 8
    if clockwise == 0:
        return AntAction.MOVE_FORWARD if not is_blocked(perception) else random_turn()
    if clockwise == 4:
        return random_turn()
    return AntAction.TURN_RIGHT if clockwise <= 4 else AntAction.TURN_LEFT


def random_turn():
    """Pick a random turn."""
    return random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])


def direction_to_closest(perception, terrain_type):
    """Return the direction to the closest visible terrain of a given type."""
    best_direction = None
    best_distance = float("inf")
    for (dx, dy), terrain in perception.visible_cells.items():
        if terrain != terrain_type or (dx, dy) == (0, 0):
            continue
        distance = math.hypot(dx, dy)
        if distance < best_distance:
            best_distance = distance
            best_direction = direction_from_delta(dx, dy)
    return best_direction


def distance_to_closest(perception, terrain_type):
    """Return the distance to the closest visible terrain of a given type."""
    best_distance = float("inf")
    for (dx, dy), terrain in perception.visible_cells.items():
        if terrain == terrain_type and (dx, dy) != (0, 0):
            best_distance = min(best_distance, math.hypot(dx, dy))
    return best_distance


def open_directions(perception):
    """List immediate non-blocked directions."""
    return [direction for direction in Direction if not is_blocked(perception, direction)]


def visible_free_distance(perception, direction, max_steps=7):
    """Count visible free cells in a straight line."""
    dx, dy = delta(direction)
    count = 0
    for step in range(1, max_steps + 1):
        terrain = perception.visible_cells.get((dx * step, dy * step))
        if terrain is None or terrain == TerrainType.WALL:
            break
        count += 1
    return count


class SmartStrategy(AntStrategy):
    """Smart role strategy: 10% carriers near colony, 90% far explorers."""

    def __init__(self):
        self.positions = {}
        self.last_actions = {}
        self.outbound_paths = {}
        self.return_paths = {}
        self.food_paths = {}
        self.food_path_indices = {}
        self.food_memory = {}
        self.last_food_taken = {}

        self.known_maps = {}
        self.visit_counts = {}
        self.recent_positions = {}
        self.gateways = {}
        self.exhausted_zones = {}
        self.avoid_zones = {}
        self.pheromone_spots = {}

        self.carrying_steps = {}
        self.gateway_marking_steps = {}
        self.gateway_marking_targets = {}
        self.gateway_probes = {}
        self.gateway_searches = {}

    def decide_action(self, perception: AntPerception) -> AntAction:
        """Update personal memory and execute the role-aware priority stack."""
        self._update_position(perception)
        self._update_known_map(perception)
        self._update_pheromone_spots(perception)
        self._remember_exploration_state(perception)
        self._update_carrying_state(perception)
        self._clear_depleted_food_memory(perception)

        terrain = current_terrain(perception)
        if not perception.has_food and terrain == TerrainType.FOOD:
            self._remember_food(perception)
            return self._remember_action(perception, AntAction.PICK_UP_FOOD)

        if perception.has_food and terrain == TerrainType.COLONY:
            self._reset_after_drop(perception)
            return self._remember_action(perception, AntAction.DROP_FOOD)

        action = self._carry_food(perception) if perception.has_food else self._search_food(perception)
        return self._remember_action(perception, action)

    def _role(self, ant_id):
        """Assign 1/10 carrier, 9/10 explorer."""
        return ROLE_CARRIER if (ant_id or 0) % 10 == 0 else ROLE_EXPLORER

    def _carry_food(self, perception):
        """Bring food home with no exploration randomness."""
        colony = direction_to_closest(perception, TerrainType.COLONY)
        if colony is not None:
            return turn_towards(perception, colony)

        route_direction = None
        mapped_home = self._mapped_direction_to(perception.ant_id, (0, 0))
        if mapped_home is not None:
            if not is_blocked(perception, mapped_home):
                route_direction = mapped_home

            else:
                join_reverse = self._direction_to_reverse_path(perception)
                if join_reverse is not None:
                    route_direction = join_reverse

        if route_direction is None:
            route_direction = self._return_path_direction(perception)

        if route_direction is None:
            route_direction = self._home_direction(perception)

        if self._should_deposit_food(perception) and self._should_mark_food_return(perception, route_direction):
            return AntAction.DEPOSIT_FOOD_PHEROMONE

        return self._move_home_or_turn(perception, route_direction)

    def _search_food(self, perception):
        """Priority: visible food, known food path, pheromone path, gateway marking, role, exploration."""
        visible_food = direction_to_closest(perception, TerrainType.FOOD)
        if visible_food is not None:
            return turn_towards(perception, visible_food)

        known_food = self._remembered_food_action(perception)
        if known_food is not None:
            return known_food

        probe = self._gateway_probe_action(perception)
        if probe is not None:
            return probe

        if self._is_marking_gateway(perception):
            if self._should_deposit_home(perception):
                return AntAction.DEPOSIT_HOME_PHEROMONE
            return self._gateway_return_action(perception)

        if self._should_deposit_home(perception):
            return AntAction.DEPOSIT_HOME_PHEROMONE

        food_pheromone = self._pheromone_path_direction(perception, perception.food_pheromone, threshold=8.0)
        if food_pheromone is not None:
            return self._move_or_escape(perception, food_pheromone)

        gateway_search = self._gateway_search_action(perception)
        if gateway_search is not None:
            return gateway_search

        if self._is_stagnating(perception):
            self._mark_current_area_to_avoid(perception)
            return self._start_unstuck(perception)

        home_pheromone = self._home_trail_outward_direction(perception, threshold=12.0)
        if home_pheromone is not None:
            return self._move_or_escape(perception, home_pheromone)

        role_direction = self._role_direction(perception)
        if role_direction is not None:
            return self._move_or_escape(perception, role_direction)

        if is_blocked(perception):
            return self._wall_escape_action(perception)

        return self._best_open_action(perception, target=self._exploration_sector(perception), force_exploration=True)

    def _role_direction(self, perception):
        """Execute role behavior after all food/path priorities."""
        if self._role(perception.ant_id) == ROLE_CARRIER:
            return self._carrier_patrol_direction(perception)

        gateway = self._new_gateway_direction(perception)
        if gateway is not None:
            return gateway
        return None

    def _carrier_patrol_direction(self, perception):
        """Keep carriers near colony to catch food pheromone trails early."""
        x, y = self.positions.get(perception.ant_id, (0, 0))
        distance = math.hypot(x, y)
        if distance > 16:
            return self._home_direction(perception)
        if distance < 5:
            return self._exploration_sector(perception, offset=2)

        radial = direction_from_delta(x, y)
        side = 2 if (perception.ant_id or 0) % 2 == 0 else -2
        return Direction((radial.value + side) % 8)

    def _should_deposit_food(self, perception):
        """Create food pheromone paths without freezing carriers."""
        carried_steps = self.carrying_steps.get(perception.ant_id, 0)
        if carried_steps == 1:
            return True
        interval = 3 if self._role(perception.ant_id) == ROLE_CARRIER else 4
        return carried_steps > 1 and (perception.steps_taken + (perception.ant_id or 0)) % interval == 0

    def _should_deposit_food_on_known_path(self, perception):
        """Mark food trails while returning from colony to a remembered food source."""
        if perception.has_food:
            return False
        if perception.ant_id not in self.food_paths:
            return False
        if not self._is_on_direct_food_colony_line(perception):
            return False
        return (perception.steps_taken + (perception.ant_id or 0)) % 4 == 0

    def _is_aligned_for_return(self, perception, direction):
        """Allow marking only once the ant is already facing its return route."""
        if direction is None:
            return False
        return perception.direction == direction and not is_blocked(perception, direction)

    def _is_aligned_for_direct_home(self, perception):
        """Allow food marking only on the direct food-colony line toward colony."""
        home = self._home_direction(perception)
        if home is None:
            return False
        return self._is_aligned_for_return(perception, home) and self._is_on_direct_food_colony_line(perception)

    def _should_mark_food_return(self, perception, route_direction):
        """Mark direct nearby food routes, and any real return route for far food."""
        if self._is_aligned_for_direct_home(perception):
            return True
        if not self._food_source_is_far(perception, minimum_distance=20):
            return False
        return self._is_aligned_for_return(perception, route_direction)

    def _food_source_is_far(self, perception, minimum_distance):
        """Tell whether the last food pickup is far enough from the colony to mark detours."""
        food = self.last_food_taken.get(perception.ant_id) or self.food_memory.get(perception.ant_id)
        return food is not None and math.hypot(food[0], food[1]) > minimum_distance

    def _is_on_direct_food_colony_line(self, perception, tolerance=1.6):
        """Check whether the ant is close to the segment colony-last-food."""
        food = self.last_food_taken.get(perception.ant_id) or self.food_memory.get(perception.ant_id)
        if food is None:
            return False

        current = self.positions.get(perception.ant_id, (0, 0))
        food_length = math.hypot(food[0], food[1])
        if food_length == 0:
            return False

        projection = (current[0] * food[0] + current[1] * food[1]) / (food_length * food_length)
        if projection < -0.05 or projection > 1.05:
            return False

        distance_to_line = abs(current[0] * food[1] - current[1] * food[0]) / food_length
        return distance_to_line <= tolerance

    def _should_deposit_home(self, perception):
        """Deposit home pheromones only while marking a useful gateway route."""
        if perception.has_food or perception.can_see_colony():
            return False
        if not self._is_marking_gateway(perception):
            return False
        return (perception.steps_taken + (perception.ant_id or 0)) % 3 == 0

    def _is_marking_gateway(self, perception):
        """Tell whether an ant is currently returning home after a gateway discovery."""
        current = self.positions.get(perception.ant_id, (0, 0))
        if perception.can_see_colony() or math.hypot(*current) <= 2:
            self.gateway_marking_steps.pop(perception.ant_id, None)
            self.gateway_marking_targets.pop(perception.ant_id, None)
            self.gateway_probes.pop(perception.ant_id, None)
            self.gateway_searches.pop(perception.ant_id, None)
            return False
        remaining = self.gateway_marking_steps.get(perception.ant_id, 0)
        return remaining > 0 or perception.ant_id in self.gateway_marking_targets

    def _gateway_marking_direction(self, perception):
        """Return to the colony after finding a useful gateway."""
        if self._is_stagnating(perception):
            self._mark_current_area_to_avoid(perception)
            return self._best_home_direction(perception, self._home_direction(perception))

        ant_id = perception.ant_id
        self.gateway_marking_steps[ant_id] = self.gateway_marking_steps.get(ant_id, 1) - 1
        mapped_home = self._mapped_direction_to(perception.ant_id, (0, 0))
        home = mapped_home or self._home_direction(perception)
        if home is not None:
            return home
        return self._mapped_random_direction(perception, target=self._home_direction(perception))

    def _gateway_return_action(self, perception):
        """Return from a gateway while avoiding wall-hugging traps."""
        direction = self._gateway_marking_direction(perception)
        if direction is None:
            return self._wall_escape_action(perception, self._home_direction(perception))
        if not is_blocked(perception, direction):
            return turn_towards(perception, direction)

        detour = self._mapped_random_direction(perception, target=self._home_direction(perception))
        if detour is not None:
            return turn_towards(perception, detour)
        return self._move_home_or_turn(perception, direction)

    def _gateway_probe_action(self, perception):
        """Verify a suspected wall gap before marking it as a gateway."""
        ant_id = perception.ant_id
        probe = self.gateway_probes.get(ant_id)
        if not probe or perception.has_food:
            return None

        current = self.positions.get(ant_id, (0, 0))
        target = probe["target"]
        crossing = probe["direction"]

        if probe["steps"] <= 0:
            self.gateway_probes.pop(ant_id, None)
            return None
        probe["steps"] -= 1

        distance = math.hypot(target[0] - current[0], target[1] - current[1])
        if distance > 1.5:
            return self._move_or_escape(perception, direction_from_delta(target[0] - current[0], target[1] - current[1]))

        if is_blocked(perception, crossing):
            self.gateway_probes.pop(ant_id, None)
            return self._wall_escape_action(perception, self._home_direction(perception))

        dx, dy = delta(crossing)
        passed_depth = (current[0] - target[0]) * dx + (current[1] - target[1]) * dy
        if passed_depth >= 1 and (passed_depth >= 2 or visible_free_distance(perception, crossing, max_steps=3) >= 2):
            self._confirm_gateway(ant_id, target)
            return self._gateway_return_action(perception)

        return turn_towards(perception, crossing)

    def _gateway_search_action(self, perception):
        """Follow an observed wall deterministically until a gap probe starts."""
        ant_id = perception.ant_id
        search = self.gateway_searches.get(ant_id)
        if not search or perception.has_food:
            return None
        if ant_id in self.gateway_probes or ant_id in self.gateway_marking_targets:
            return None
        if search["steps"] <= 0:
            self.gateway_searches.pop(ant_id, None)
            return None

        search["steps"] -= 1
        direction = search["direction"]
        if is_blocked(perception, direction):
            direction = Direction((direction.value + 4) % 8)
            search["direction"] = direction
            if is_blocked(perception, direction):
                return self._best_open_action(perception, target=self._exploration_sector(perception), force_exploration=True)
        return turn_towards(perception, direction)

    def _confirm_gateway(self, ant_id, target):
        """Store a verified gateway and start home-route marking."""
        self.gateways.setdefault(ant_id, set()).add(target)
        self.gateway_marking_targets[ant_id] = target
        self.gateway_marking_steps[ant_id] = 300
        self.gateway_probes.pop(ant_id, None)
        self.gateway_searches.pop(ant_id, None)

    def _home_trail_outward_direction(self, perception, threshold):
        """Follow home pheromones away from the colony, not back into it."""
        current = self.positions.get(perception.ant_id, (0, 0))
        candidates = []
        for direction in open_directions(perception):
            dx, dy = delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            if math.hypot(*next_pos) <= math.hypot(*current):
                continue

            score = 0.0
            for (pdx, pdy), value in perception.home_pheromone.items():
                if value < threshold:
                    continue
                absolute = (current[0] + pdx, current[1] + pdy)
                outward = direction_from_delta(absolute[0], absolute[1])
                angle = angular_distance(direction, outward)
                if angle <= 2:
                    score += value * ((3 - angle) / 3) / max(1.0, math.hypot(pdx, pdy))
            if score >= threshold:
                candidates.append((score + math.hypot(*next_pos) * 0.05, direction))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _pheromone_path_direction(self, perception, pheromone_map, threshold):
        """Follow pheromone trails in the colony-to-pheromone direction."""
        current = self.positions.get(perception.ant_id, (0, 0))
        candidates = []
        for direction in open_directions(perception):
            dx, dy = delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            if self._inside_exhausted_zone(perception.ant_id, next_pos):
                continue

            score = 0.0
            for (pdx, pdy), value in pheromone_map.items():
                if value < threshold or (pdx, pdy) == (0, 0):
                    continue
                absolute = (current[0] + pdx, current[1] + pdy)
                colony_to_pheromone = direction_from_delta(absolute[0], absolute[1])
                angle = angular_distance(direction, colony_to_pheromone)
                if angle > 2:
                    continue
                score += value * ((3 - angle) / 3) / max(1.0, math.hypot(pdx, pdy))

            if score >= threshold:
                if direction == perception.direction:
                    score *= 1.08
                candidates.append((score, direction))

        if not candidates:
            return None
        best = max(candidates, key=lambda item: item[0])[0]
        useful = [(score, direction) for score, direction in candidates if score >= best * 0.65]
        return random.choices([direction for _, direction in useful], weights=[score for score, _ in useful], k=1)[0]

    def _remember_food(self, perception):
        """Remember food location and the path to exploit it later."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        position = self.positions.get(ant_id, (0, 0))
        self.food_memory[ant_id] = position
        self.last_food_taken[ant_id] = position
        self._clear_exhausted_zone_near(ant_id, position)
        self._append_path_position(ant_id, position)
        path = list(self.outbound_paths.get(ant_id, [(0, 0)]))
        self.food_paths[ant_id] = path
        self.food_path_indices[ant_id] = 0
        self.return_paths[ant_id] = list(reversed(path))

    def _remembered_food_action(self, perception):
        """Exploit remembered food until the local search declares it empty."""
        ant_id = perception.ant_id
        food = self.food_memory.get(ant_id)

        if food is None or self._inside_exhausted_zone(ant_id, food):
            self._forget_food(ant_id)
            return None

        direction = self._food_path_direction(perception)
        if direction is None and perception.steps_taken % 8 == 0:
            direction = self._mapped_direction_to(ant_id, food)
        if direction is None:
            direction = direction_from_delta(food[0] - self.positions.get(ant_id, (0, 0))[0], food[1] - self.positions.get(ant_id, (0, 0))[1])
        if direction is None:
            return None
        if self._should_deposit_food_on_known_path(perception) and self._is_aligned_for_return(perception, direction):
            return AntAction.DEPOSIT_FOOD_PHEROMONE
        return self._move_or_escape(perception, direction)

    def _food_path_direction(self, perception):
        """Follow the personal colony-to-food path."""
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
        if target == current:
            return None
        return direction_from_delta(target[0] - current[0], target[1] - current[1])

    def _return_path_direction(self, perception):
        """Follow the remembered food-to-colony path."""
        ant_id = perception.ant_id
        path = self.return_paths.get(ant_id)
        if not path:
            return None
        current = self.positions.get(ant_id, (0, 0))
        nearest = min(range(len(path)), key=lambda i: math.hypot(path[i][0] - current[0], path[i][1] - current[1]))
        target_index = min(nearest + 1, len(path) - 1) if math.hypot(path[nearest][0] - current[0], path[nearest][1] - current[1]) <= 1.5 else nearest
        target = path[target_index]
        if target == current:
            return None
        return direction_from_delta(target[0] - current[0], target[1] - current[1])

    def _direction_to_reverse_path(self, perception):
        """Move toward the closest point of the remembered reverse path."""
        path = self.return_paths.get(perception.ant_id)
        if not path:
            return None

        current = self.positions.get(perception.ant_id, (0, 0))
        target = min(path, key=lambda point: math.hypot(point[0] - current[0], point[1] - current[1]))
        if target == current:
            return self._return_path_direction(perception)
        return direction_from_delta(target[0] - current[0], target[1] - current[1])

    def _clear_depleted_food_memory(self, perception):
        """Mark food zones empty after searching just beyond the last taken food."""
        ant_id = perception.ant_id
        if ant_id is None or perception.has_food or perception.can_see_food():
            return
        food = self.last_food_taken.get(ant_id)
        if food is None:
            return

        current = self.positions.get(ant_id, (0, 0))
        colony_to_food = direction_from_delta(food[0], food[1])
        fx, fy = delta(colony_to_food)
        search_points = [(food[0], food[1])]
        for step in (1, 2, 3):
            search_points.append((food[0] + fx * step, food[1] + fy * step))
            search_points.append((food[0] + fx * step + fy, food[1] + fy * step - fx))
            search_points.append((food[0] + fx * step - fy, food[1] + fy * step + fx))

        if min(math.hypot(px - current[0], py - current[1]) for px, py in search_points) > 1.5:
            return
        self._remember_exhausted_zone(ant_id, food, radius=4)
        self._forget_food(ant_id)

    def _forget_food(self, ant_id):
        """Forget a depleted food target."""
        self.food_memory.pop(ant_id, None)
        self.food_paths.pop(ant_id, None)
        self.food_path_indices.pop(ant_id, None)

    def _home_direction(self, perception):
        """Return the direction to the known colony origin."""
        x, y = self.positions.get(perception.ant_id, (0, 0))
        if x == 0 and y == 0:
            return None
        return direction_from_delta(-x, -y)

    def _move_or_escape(self, perception, direction):
        """Move toward a direction, choosing an open alternative around walls."""
        if direction is None:
            return random_turn()
        if is_blocked(perception) or is_blocked(perception, direction):
            return self._wall_escape_action(perception, direction)
        return turn_towards(perception, direction)

    def _move_home_or_turn(self, perception, direction):
        """Move toward home without exploration-biased randomness."""
        if direction is None:
            direction = self._home_direction(perception)
        if direction is None:
            return AntAction.MOVE_FORWARD if not is_blocked(perception) else random_turn()
        if is_blocked(perception) or is_blocked(perception, direction):
            direction = self._best_home_direction(perception, direction)
        return turn_towards(perception, direction)

    def _best_home_direction(self, perception, preferred):
        """Pick the open direction that most reduces distance to colony."""
        directions = open_directions(perception)
        if not directions:
            return None

        current = self.positions.get(perception.ant_id, (0, 0))

        def score(direction):
            dx, dy = delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            value = -math.hypot(*next_pos)
            if preferred is not None:
                value -= angular_distance(direction, preferred) * 0.35
            if direction == perception.direction:
                value += 0.1
            value += visible_free_distance(perception, direction, max_steps=3) * 0.03
            return value

        return max(directions, key=score)

    def _wall_escape_action(self, perception, preferred=None):
        """Pick a scored open direction around a blockage."""
        if random.random() < 0.5:
            target = self._home_direction(perception)
        else:
            target = preferred
        direction = self._mapped_random_direction(perception, target=target)
        if direction is None:
            direction = self._best_open_direction(perception, target=preferred, force_exploration=True)
        return turn_towards(perception, direction) if direction is not None else random_turn()

    def _best_open_action(self, perception, target=None, carrying_food=False, force_exploration=False):
        """Turn toward the best immediate open direction."""
        return turn_towards(perception, self._best_open_direction(perception, target, carrying_food, force_exploration))

    def _best_open_direction(self, perception, target=None, carrying_food=False, force_exploration=False):
        """Score open directions with personal memory and exploration pressure."""
        directions = open_directions(perception)
        if not directions:
            return None
        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        role = self._role(ant_id)

        def score(direction):
            dx, dy = delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            value = random.random() * 0.2
            value += visible_free_distance(perception, direction) * 0.1
            if direction == perception.direction:
                value += 0.35
            if target is not None:
                value -= angular_distance(direction, target) * 0.55
            if carrying_food:
                value -= math.hypot(*next_pos) * 0.08
            else:
                value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 8) * 0.3
                value += self._frontier_score(ant_id, direction)
                if self._inside_exhausted_zone(ant_id, next_pos) or self._inside_avoid_zone(ant_id, next_pos):
                    value -= 3.0
                if role == ROLE_EXPLORER:
                    value += math.hypot(*next_pos) * 0.035
            return value

        return max(directions, key=score)

    def _exploration_sector(self, perception, offset=0):
        """Give each ant a stable exploration sector."""
        sectors = [(1, -1), (-1, -1), (1, 1), (-1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = sectors[((perception.ant_id or 0) + offset) % len(sectors)]
        return direction_from_delta(dx, dy)

    def _new_gateway_direction(self, perception):
        """Target remembered gateways that are not already saturated."""
        current = self.positions.get(perception.ant_id, (0, 0))
        gateways = self.gateways.get(perception.ant_id, set())
        if not gateways:
            return None
        target = min(gateways, key=lambda g: self.visit_counts.get(perception.ant_id, {}).get(g, 0) + math.hypot(g[0] - current[0], g[1] - current[1]) * 0.1)
        if math.hypot(target[0] - current[0], target[1] - current[1]) < 2:
            return None
        return direction_from_delta(target[0] - current[0], target[1] - current[1])

    def _near_gateway(self, perception):
        """Check whether the ant is on/near a remembered gateway."""
        current = self.positions.get(perception.ant_id, (0, 0))
        return any(math.hypot(g[0] - current[0], g[1] - current[1]) <= 2 for g in self.gateways.get(perception.ant_id, set()))

    def _update_position(self, perception):
        """Update home-relative odometry from the previous forward move."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        self.positions.setdefault(ant_id, (0, 0))
        self.outbound_paths.setdefault(ant_id, [(0, 0)])
        last = self.last_actions.get(ant_id)
        if last is None:
            return
        action, direction, expected = last
        if action != AntAction.MOVE_FORWARD or not expected:
            return
        x, y = self.positions[ant_id]
        dx, dy = delta(direction)
        position = (x + dx, y + dy)
        self.positions[ant_id] = position
        if not perception.has_food:
            self._append_path_position(ant_id, position)

    def _remember_action(self, perception, action):
        """Store action state needed for next-step odometry."""
        if perception.ant_id is not None:
            expected = action == AntAction.MOVE_FORWARD and not is_blocked(perception)
            self.last_actions[perception.ant_id] = (action, perception.direction, expected)
        return action

    def _append_path_position(self, ant_id, position):
        """Append to the personal outbound path."""
        path = self.outbound_paths.setdefault(ant_id, [(0, 0)])
        if not path or path[-1] != position:
            path.append(position)
        if len(path) > MAX_PATH_LENGTH:
            del path[: len(path) - MAX_PATH_LENGTH]

    def _reset_after_drop(self, perception):
        """Reset colony-relative state after a successful drop."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        self.positions[ant_id] = (0, 0)
        self.outbound_paths[ant_id] = [(0, 0)]
        self.return_paths.pop(ant_id, None)
        self.carrying_steps.pop(ant_id, None)
        self.recent_positions[ant_id] = [(0, 0)]
        if ant_id in self.food_paths:
            self.food_path_indices[ant_id] = 0

    def _update_carrying_state(self, perception):
        """Count how long an ant has carried food."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        if perception.has_food:
            self.carrying_steps[ant_id] = self.carrying_steps.get(ant_id, 0) + 1
        else:
            self.carrying_steps.pop(ant_id, None)

    def _remember_exploration_state(self, perception):
        """Update visits, recent positions, and loop avoid zones."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        position = self.positions.get(ant_id, (0, 0))
        visits = self.visit_counts.setdefault(ant_id, {})
        visits[position] = visits.get(position, 0) + 1
        if len(visits) > MAX_VISITED_CELLS:
            self._trim_map_around(visits, position, MAX_VISITED_CELLS)

        recent = self.recent_positions.setdefault(ant_id, [])
        recent.append(position)
        if len(recent) > RECENT_WINDOW:
            del recent[: len(recent) - RECENT_WINDOW]
        if len(recent) == RECENT_WINDOW:
            center_x = sum(x for x, _ in recent) / len(recent)
            center_y = sum(y for _, y in recent) / len(recent)
            spread = max(math.hypot(x - center_x, y - center_y) for x, y in recent)
            if spread <= 4 and len(set(recent)) <= 9:
                zones = self.avoid_zones.setdefault(ant_id, [])
                zones.append((int(round(center_x)), int(round(center_y)), 6))
                if len(zones) > 8:
                    del zones[: len(zones) - 8]

    def _update_known_map(self, perception):
        """Merge visible terrain and detect personal gateways."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        current = self.positions.get(ant_id, (0, 0))
        known = self.known_maps.setdefault(ant_id, {})
        saw_wall = TerrainType.WALL in perception.visible_cells.values()
        if saw_wall and not perception.has_food:
            self._start_gateway_search(perception, current)
        detect_gateway = (
            saw_wall
            and not perception.has_food
            and (
                ant_id in self.gateway_searches
                or perception.steps_taken % GATEWAY_DETECTION_PERIOD == (ant_id or 0) % GATEWAY_DETECTION_PERIOD
            )
        )
        for (dx, dy), terrain in perception.visible_cells.items():
            position = (current[0] + dx, current[1] + dy)
            known[position] = terrain
            if detect_gateway and terrain != TerrainType.WALL:
                if math.hypot(dx, dy) > 3:
                    continue
                gateways = self.gateways.setdefault(ant_id, set())
                probe = self._gateway_probe(known, position, current)
                if probe is not None:
                    target, probe_direction = probe
                    self._start_gateway_probe(ant_id, current, target, probe_direction)
                elif position in gateways and self._is_dead_end(known, position):
                    gateways.discard(position)
        if len(known) > MAX_KNOWN_CELLS:
            self._trim_map_around(known, current, MAX_KNOWN_CELLS)
        self._trim_gateways(ant_id, current)

    def _start_gateway_search(self, perception, current):
        """Start a deterministic wall scan after a wall has been observed."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        if ant_id in self.gateway_probes or ant_id in self.gateway_marking_targets:
            return
        if perception.can_see_food() or any(value >= 8.0 for value in perception.food_pheromone.values()):
            return
        if math.hypot(*current) < 10:
            return
        existing = self.gateway_searches.get(ant_id)
        if existing is not None and existing["steps"] > 0:
            return

        wall_direction = self._closest_wall_direction(perception)
        if wall_direction is None:
            return
        if distance_to_closest(perception, TerrainType.WALL) > 4:
            return
        if wall_direction in (Direction.EAST, Direction.WEST):
            choices = (Direction.NORTH, Direction.SOUTH)
        elif wall_direction in (Direction.NORTH, Direction.SOUTH):
            choices = (Direction.EAST, Direction.WEST)
        elif wall_direction in (Direction.NORTHEAST, Direction.SOUTHWEST):
            choices = (Direction.NORTHWEST, Direction.SOUTHEAST)
        else:
            choices = (Direction.NORTHEAST, Direction.SOUTHWEST)

        direction = choices[(ant_id or 0) % 2]
        if is_blocked(perception, direction):
            direction = choices[1 - ((ant_id or 0) % 2)]
        self.gateway_searches[ant_id] = {
            "direction": direction,
            "steps": 18,
        }

    def _closest_wall_direction(self, perception):
        """Return the direction to the closest visible wall."""
        best = None
        best_distance = float("inf")
        for (dx, dy), terrain in perception.visible_cells.items():
            if terrain != TerrainType.WALL or (dx, dy) == (0, 0):
                continue
            distance = math.hypot(dx, dy)
            if distance < best_distance:
                best_distance = distance
                best = direction_from_delta(dx, dy)
        return best

    def _looks_like_gateway(self, known, position):
        """Detect a passable opening through a wall/river-like barrier."""
        if self._is_dead_end(known, position):
            return False
        return self._vertical_barrier_gateway(known, position) or self._horizontal_barrier_gateway(known, position)

    def _gateway_probe_direction(self, known, position, current):
        """Return the crossing direction to test a suspected wall opening."""
        probe = self._gateway_probe(known, position, current)
        return probe[1] if probe is not None else None

    def _gateway_probe(self, known, position, current):
        """Return the target gap cell and crossing direction to test it."""
        if self._is_tight_dead_end(known, position):
            return None
        if self._vertical_gap_candidate(known, position):
            return position, Direction.EAST if current[0] <= position[0] else Direction.WEST
        if self._horizontal_gap_candidate(known, position):
            return position, Direction.SOUTH if current[1] <= position[1] else Direction.NORTH
        adjacent = self._adjacent_river_gap_probe(known, position, current)
        if adjacent is not None:
            return adjacent
        return None

    def _start_gateway_probe(self, ant_id, current, target, direction):
        """Start pursuing an observed wall gap until it can be tested."""
        if ant_id in self.gateway_marking_targets:
            return
        existing = self.gateway_probes.get(ant_id)
        if existing is not None:
            old = existing["target"]
            old_distance = math.hypot(old[0] - current[0], old[1] - current[1])
            new_distance = math.hypot(target[0] - current[0], target[1] - current[1])
            if old_distance <= new_distance:
                return
        if math.hypot(target[0] - current[0], target[1] - current[1]) > 7:
            return
        self.gateway_probes[ant_id] = {
            "target": target,
            "direction": direction,
            "steps": 45,
        }

    def _vertical_barrier_gateway(self, known, position):
        """Detect a bridge through a mostly vertical wall line."""
        x, y = position
        if not self._not_wall(known, (x - 1, y)) or not self._not_wall(known, (x + 1, y)):
            return False
        north_wall = self._wall_seen_along(known, x, y, 0, -1)
        south_wall = self._wall_seen_along(known, x, y, 0, 1)
        if not (north_wall and south_wall):
            return False
        return self._open_run(known, x, y, -1, 0) >= 1 and self._open_run(known, x, y, 1, 0) >= 1

    def _vertical_gap_candidate(self, known, position):
        """Detect a possible opening in a vertical barrier before full confirmation."""
        x, y = position
        if not self._not_wall(known, (x - 1, y)) or not self._not_wall(known, (x + 1, y)):
            return False
        wall_evidence = self._wall_seen_along(known, x, y, 0, -1) or self._wall_seen_along(known, x, y, 0, 1)
        return wall_evidence and self._open_run(known, x, y, -1, 0) >= 1 and self._open_run(known, x, y, 1, 0) >= 1

    def _horizontal_barrier_gateway(self, known, position):
        """Detect a bridge through a mostly horizontal wall line."""
        x, y = position
        if not self._not_wall(known, (x, y - 1)) or not self._not_wall(known, (x, y + 1)):
            return False
        west_wall = self._wall_seen_along(known, x, y, -1, 0)
        east_wall = self._wall_seen_along(known, x, y, 1, 0)
        if not (west_wall and east_wall):
            return False
        return self._open_run(known, x, y, 0, -1) >= 1 and self._open_run(known, x, y, 0, 1) >= 1

    def _horizontal_gap_candidate(self, known, position):
        """Detect a possible opening in a horizontal barrier before full confirmation."""
        x, y = position
        if not self._not_wall(known, (x, y - 1)) or not self._not_wall(known, (x, y + 1)):
            return False
        wall_evidence = self._wall_seen_along(known, x, y, -1, 0) or self._wall_seen_along(known, x, y, 1, 0)
        return wall_evidence and self._open_run(known, x, y, 0, -1) >= 1 and self._open_run(known, x, y, 0, 1) >= 1

    def _adjacent_river_gap_probe(self, known, position, current):
        """Detect a river gap from a cell beside the wall line."""
        x, y = position
        if self._river_gap_at(known, x + 1, y, vertical=True):
            return (x + 1, y), Direction.EAST
        if self._river_gap_at(known, x - 1, y, vertical=True):
            return (x - 1, y), Direction.WEST
        if self._river_gap_at(known, x, y + 1, vertical=False):
            return (x, y + 1), Direction.SOUTH
        if self._river_gap_at(known, x, y - 1, vertical=False):
            return (x, y - 1), Direction.NORTH
        return None

    def _river_gap_at(self, known, x, y, vertical):
        """Recognize an opening inside a long straight barrier."""
        if not self._known_free(known, (x, y)):
            return False
        if vertical:
            before = self._wall_seen_along(known, x, y, 0, -1)
            after = self._wall_seen_along(known, x, y, 0, 1)
            wall_count = self._wall_count_along(known, x, y, 0, -1) + self._wall_count_along(known, x, y, 0, 1)
            sides_open = self._not_wall(known, (x - 1, y)) and self._not_wall(known, (x + 1, y))
        else:
            before = self._wall_seen_along(known, x, y, -1, 0)
            after = self._wall_seen_along(known, x, y, 1, 0)
            wall_count = self._wall_count_along(known, x, y, -1, 0) + self._wall_count_along(known, x, y, 1, 0)
            sides_open = self._not_wall(known, (x, y - 1)) and self._not_wall(known, (x, y + 1))
        return sides_open and (before or after) and wall_count >= 1

    def _not_wall(self, known, position):
        """Accept known free cells and still-unseen cells, but never walls."""
        return known.get(position) != TerrainType.WALL

    def _known_free(self, known, position):
        """Treat any known non-wall cell as traversable for gateway detection."""
        terrain = known.get(position)
        return terrain is not None and terrain != TerrainType.WALL

    def _wall_seen_along(self, known, x, y, dx, dy):
        """Look for barrier evidence in a straight line from a candidate opening."""
        for step in range(1, GATEWAY_SCAN_RADIUS + 1):
            terrain = known.get((x + dx * step, y + dy * step))
            if terrain == TerrainType.WALL:
                return True
        return False

    def _wall_count_along(self, known, x, y, dx, dy):
        """Count wall evidence in a straight line from a candidate opening."""
        count = 0
        for step in range(1, GATEWAY_SCAN_RADIUS + 1):
            if known.get((x + dx * step, y + dy * step)) == TerrainType.WALL:
                count += 1
        return count

    def _free_run(self, known, x, y, dx, dy):
        """Count known free cells from a candidate gateway in one direction."""
        count = 0
        for step in range(1, 4):
            terrain = known.get((x + dx * step, y + dy * step))
            if terrain == TerrainType.WALL:
                break
            if terrain is not None:
                count += 1
        return count

    def _open_run(self, known, x, y, dx, dy):
        """Count cells from a gateway that are not known walls."""
        count = 0
        for step in range(1, 4):
            terrain = known.get((x + dx * step, y + dy * step))
            if terrain == TerrainType.WALL:
                break
            count += 1
        return count

    def _is_dead_end(self, known, position):
        """Reject gateway candidates that look like local cul-de-sacs."""
        open_count = 0
        unknown_count = 0
        wall_count = 0
        for direction in Direction:
            dx, dy = delta(direction)
            terrain = known.get((position[0] + dx, position[1] + dy))
            if terrain is None:
                unknown_count += 1
            elif terrain == TerrainType.WALL:
                wall_count += 1
            else:
                open_count += 1
        if open_count <= 1:
            return True
        return open_count <= 2 and unknown_count == 0 and wall_count >= 5

    def _is_tight_dead_end(self, known, position):
        """Reject only obvious one-way pockets before probing a possible gap."""
        open_count = 0
        for direction in Direction:
            dx, dy = delta(direction)
            if self._not_wall(known, (position[0] + dx, position[1] + dy)):
                open_count += 1
        return open_count <= 1

    def _trim_gateways(self, ant_id, current):
        """Keep the nearest gateway memories."""
        gateways = self.gateways.get(ant_id)
        if not gateways or len(gateways) <= MAX_GATEWAYS:
            return
        closest = sorted(gateways, key=lambda g: math.hypot(g[0] - current[0], g[1] - current[1]))[:MAX_GATEWAYS]
        self.gateways[ant_id] = set(closest)

    def _update_pheromone_spots(self, perception):
        """Remember strong pheromone spots as personal landmarks."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        current = self.positions.get(ant_id, (0, 0))
        for kind, pheromones in (("food", perception.food_pheromone), ("home", perception.home_pheromone)):
            spots = self.pheromone_spots.setdefault(ant_id, {}).setdefault(kind, {})
            for (dx, dy), value in pheromones.items():
                if value < 35 or (dx, dy) == (0, 0):
                    continue
                position = (current[0] + dx, current[1] + dy)
                if kind == "food" and self._inside_exhausted_zone(ant_id, position):
                    continue
                spots[position] = max(spots.get(position, 0.0), value)
            if len(spots) > MAX_PHEROMONE_SPOTS:
                self.pheromone_spots[ant_id][kind] = dict(sorted(spots.items(), key=lambda item: item[1], reverse=True)[:MAX_PHEROMONE_SPOTS])

    def _remembered_pheromone_direction(self, perception, kind, closer_to_colony=False):
        """Move toward remembered pheromone landmarks."""
        spots = self.pheromone_spots.get(perception.ant_id, {}).get(kind, {})
        if not spots:
            return None
        current = self.positions.get(perception.ant_id, (0, 0))
        candidates = []
        for position, value in spots.items():
            if closer_to_colony and math.hypot(*position) >= math.hypot(*current):
                continue
            distance = math.hypot(position[0] - current[0], position[1] - current[1])
            if distance > 1:
                candidates.append((value / (1 + distance * 0.2), position))
        if not candidates:
            return None
        _, target = max(candidates, key=lambda item: item[0])
        return direction_from_delta(target[0] - current[0], target[1] - current[1])

    def _is_stagnating(self, perception):
        """Detect when an ant loops in the same small area."""
        recent = self.recent_positions.get(perception.ant_id, [])
        if len(recent) < RECENT_WINDOW:
            return False
        center_x = sum(x for x, _ in recent) / len(recent)
        center_y = sum(y for _, y in recent) / len(recent)
        spread = max(math.hypot(x - center_x, y - center_y) for x, y in recent)
        return spread <= STAGNATION_RADIUS and len(set(recent)) <= STAGNATION_UNIQUE_LIMIT

    def _mark_current_area_to_avoid(self, perception):
        """Remember a loop area so exploration scores push away from it."""
        ant_id = perception.ant_id
        if ant_id is None:
            return
        x, y = self.positions.get(ant_id, (0, 0))
        zones = self.avoid_zones.setdefault(ant_id, [])
        zones.append((x, y, 6))
        if len(zones) > 8:
            del zones[: len(zones) - 8]

    def _mapped_random_direction(self, perception, target=None):
        """Randomly choose an open direction weighted by frontier and target value."""
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        weights = []
        for direction in directions:
            dx, dy = delta(direction)
            current = self.positions.get(ant_id, (0, 0))
            next_pos = (current[0] + dx, current[1] + dy)
            weight = 1.0 + max(0.0, self._frontier_score(ant_id, direction))
            weight += max(0.0, visible_free_distance(perception, direction) * 0.08)
            weight -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 8) * 0.2
            if self._inside_avoid_zone(ant_id, next_pos):
                weight *= 0.2
            if direction == perception.direction:
                weight += 0.6
            if target is not None:
                weight += 2.0 / (1 + angular_distance(direction, target))
            weights.append(max(0.05, weight))

        return random.choices(directions, weights=weights, k=1)[0]

    def _start_unstuck(self, perception):
        """Pick one exploration-biased action away from a stagnation zone."""
        direction = self._mapped_random_direction(perception, target=self._exploration_sector(perception))
        if direction is None:
            return random_turn()
        return turn_towards(perception, direction)

    def _frontier_score(self, ant_id, direction):
        """Reward directions that reveal unknown cells."""
        known = self.known_maps.get(ant_id, {})
        current = self.positions.get(ant_id, (0, 0))
        dx, dy = delta(direction)
        score = 0.0
        for step in range(1, FRONTIER_LOOKAHEAD + 1):
            position = (current[0] + dx * step, current[1] + dy * step)
            terrain = known.get(position)
            if terrain == TerrainType.WALL:
                score -= 1.4 / step
                break
            if terrain is None:
                score += 1.2 / step
        return score

    def _mapped_direction_to(self, ant_id, target):
        """Bounded A* over known non-wall cells."""
        known = self.known_maps.get(ant_id, {})
        current = self.positions.get(ant_id, (0, 0))
        if ant_id is None or current == target or target not in known:
            return None

        queue = [(0.0, 0.0, current)]
        came_from = {current: None}
        costs = {current: 0.0}
        searched = 0
        while queue and searched < MAX_ASTAR_NODES:
            _, cost, position = heapq.heappop(queue)
            if cost > costs.get(position, float("inf")):
                continue
            searched += 1
            if position == target:
                break
            for direction in Direction:
                dx, dy = delta(direction)
                neighbor = (position[0] + dx, position[1] + dy)
                if known.get(neighbor) == TerrainType.WALL or neighbor not in known:
                    continue
                new_cost = cost + 1.0 + min(self.visit_counts.get(ant_id, {}).get(neighbor, 0), 6) * 0.1
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

    def _remember_exhausted_zone(self, ant_id, center, radius):
        """Remember a depleted food zone."""
        zones = self.exhausted_zones.setdefault(ant_id, [])
        zone = (int(round(center[0])), int(round(center[1])), radius)
        if zone not in zones:
            zones.append(zone)
        if len(zones) > 8:
            del zones[: len(zones) - 8]

    def _clear_exhausted_zone_near(self, ant_id, position):
        """Clear an exhausted marker when food is found there again."""
        zones = self.exhausted_zones.get(ant_id)
        if not zones:
            return
        self.exhausted_zones[ant_id] = [z for z in zones if math.hypot(position[0] - z[0], position[1] - z[1]) > z[2]]

    def _inside_exhausted_zone(self, ant_id, position):
        """Check exhausted zones."""
        return any(math.hypot(position[0] - x, position[1] - y) <= radius for x, y, radius in self.exhausted_zones.get(ant_id, []))

    def _inside_avoid_zone(self, ant_id, position):
        """Check local loop-avoid zones."""
        return any(math.hypot(position[0] - x, position[1] - y) <= radius for x, y, radius in self.avoid_zones.get(ant_id, []))

    def _trim_map_around(self, mapping, center, max_size):
        """Keep nearest entries in a bounded memory dictionary."""
        keep = set(sorted(mapping, key=lambda pos: math.hypot(pos[0] - center[0], pos[1] - center[1]))[:max_size])
        for position in list(mapping):
            if position not in keep:
                del mapping[position]
