import heapq
import math
import random

from ant import AntAction, AntStrategy
from common import Direction, TerrainType
from environment import AntPerception


MAX_ENVIRONMENT_CELLS = 500 * 500
MAX_PATH_LENGTH = MAX_ENVIRONMENT_CELLS
MAX_KNOWN_CELLS = MAX_ENVIRONMENT_CELLS
MAX_VISITED_CELLS = MAX_ENVIRONMENT_CELLS
MAX_MAP_SEARCH_NODES = MAX_ENVIRONMENT_CELLS

MAX_GATEWAYS = 10
MAX_TRAIL_CELLS = 4096
RECENT_WINDOW = 50
STAGNATION_RADIUS = 8
STAGNATION_UNIQUE_LIMIT = 5
FRONTIER_LOOKAHEAD = 3
TRAIL_LENGTH = 50
EXPLORE_FOOD_STEPS = 10
INVERSE_CARRIER_SEARCH_STEPS = 45
EMPTY_FOOD_ESCAPE_STEPS = 14
EMPTY_FOOD_ESCAPE_RADIUS = 10
GATEWAY_ANALYSIS_STEPS = 8
CARRIER_TRAIL_SCORE_WEIGHT = 10
CARRIER_TRAIL_EXPLORATION_WEIGHT = 1.4
CARRIER_TRAIL_MIN_ACTION_SCORE = 0.4
CARRIER_CONE_LENGTH = 18
CARRIER_CONE_WIDTH = 5
CARRIER_CONE_TTL = 45
CARRIER_CONE_WEIGHT = 10
RECENT_REVISIT_PENALTY = 0.65
SEEN_EMPTY_VISIT_WEIGHT = 1
EXPLORATION_AWAY_FROM_COLONY_WEIGHT = 0.85
EXPLORATION_MODE_PERIOD = 80
LOCAL_WALL_CONTACT_PENALTY = 0.9
RETURN_SUBSTATE_RESET_STEPS = 150
RETURN_GATEWAY_RESET_RADIUS = 1.1

STATE_INIT = "init"
STATE_EXPLORING = "exploring"
STATE_COLLECTING = "collecting"
STATE_RETURNING = "returning"
STATE_FOLLOW_FOOD_TO_FOOD = "follow_food_to_food"
STATE_INVERSE_CARRIER = "inverse_carrier"
STATE_EXPLORE_FOOD = "explore_food"
STATE_GATEWAY_ANALYSIS = "gateway_analysis"
STATE_FOLLOW_WALL = "follow_wall"

RETURN_SUBSTATE_BLOCKED = "blocked"
RETURN_SUBSTATE_STAGNATING = "stagnating"
RETURN_SUBSTATE_FOLLOW_PATH = "follow_food_to_colony"
RETURN_TROUBLE_SUBSTATES = {
    RETURN_SUBSTATE_BLOCKED,
    RETURN_SUBSTATE_STAGNATING,
    RETURN_SUBSTATE_FOLLOW_PATH,
}

EXPLORATION_LOCAL = "local"
EXPLORATION_FAR = "far"

DIRECTION_BY_STEP = {
    (0, -1): Direction.NORTH,
    (1, -1): Direction.NORTHEAST,
    (1, 0): Direction.EAST,
    (1, 1): Direction.SOUTHEAST,
    (0, 1): Direction.SOUTH,
    (-1, 1): Direction.SOUTHWEST,
    (-1, 0): Direction.WEST,
    (-1, -1): Direction.NORTHWEST,
}


def current_terrain(perception):
    return perception.visible_cells.get((0, 0))


def as_direction(direction):
    if direction is None or isinstance(direction, Direction):
        return direction
    return Direction(direction)


def direction_from_vector(dx, dy):
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    return DIRECTION_BY_STEP.get((step_x, step_y), Direction.NORTH)


def is_blocked(perception, direction=None):
    dx, dy = Direction.get_delta(as_direction(direction) or perception.direction)
    terrain = perception.visible_cells.get((dx, dy))
    return terrain is None or terrain == TerrainType.WALL


def angular_distance(direction, target):
    direction = as_direction(direction)
    target = as_direction(target)
    if direction is None or target is None:
        return 4
    diff = abs(direction.value - target.value) % 8
    return min(diff, 8 - diff)


def random_turn():
    return random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])


def selectmove(perception, target_direction):
    target_direction = as_direction(target_direction)
    if target_direction is None:
        return random_turn()

    clockwise = (target_direction.value - perception.direction.value) % 8
    if clockwise == 0:
        return AntAction.MOVE_FORWARD if not is_blocked(perception) else random_turn()
    if clockwise == 4:
        return random_turn()
    return AntAction.TURN_RIGHT if clockwise <= 4 else AntAction.TURN_LEFT


def visible_free_distance(perception, direction, max_steps=8):
    dx, dy = Direction.get_delta(direction)
    distance = 0
    for step in range(1, max_steps + 1):
        terrain = perception.visible_cells.get((dx * step, dy * step))
        if terrain is None or terrain == TerrainType.WALL:
            break
        distance += 1
    return distance


def open_directions(perception):
    return [direction for direction in Direction if not is_blocked(perception, direction)]


def can_see_wall(perception):
    return TerrainType.WALL in perception.visible_cells.values()


class NonCooperativeStrategy(AntStrategy):
    """Independent strategy: personal memory, no pheromones."""

    def __init__(self):
        self.states = {}
        self.substates = {}

        self.positions = {}
        self.last_actions = {}
        self.outbound_paths = {}
        self.return_paths = {}

        self.food_memory = {}
        self.food_colony_paths = {}
        self.stable_food_colony_paths = {}
        self.colony_food_paths = {}
        self.stable_colony_food_paths = {}

        self.known_maps = {}
        self.visit_counts = {}
        self.recent_positions = {}
        self.avoid_zones = {}
        self.gateways = {}

        self.carrier_directions = {}
        self.carrier_trails = {}
        self.carrier_cones = {}
        self.previous_unladen_neighbors = {}
        self.food_explore_steps = {}
        self.food_search_targets = {}
        self.inverse_carrier_steps = {}
        self.empty_food_escape_steps = {}
        self.empty_food_escape_targets = {}
        self.gateway_analysis_steps = {}
        self.return_reset_steps = {}
        self.return_last_gateway = {}

        self.seen_ants = set()
        self.initial_direction_counts = {}
        self.sidestep_plans = {}

    def _state(self, ant_id):
        return self.states.get(ant_id, STATE_INIT)

    def _set_state(self, ant_id, state):
        if ant_id is not None:
            self.states[ant_id] = state

    def _substate(self, ant_id):
        return self.substates.get(ant_id)

    def _set_substate(self, ant_id, substate):
        if ant_id is None:
            return
        if substate is None:
            self.substates.pop(ant_id, None)
        else:
            self.substates[ant_id] = substate

    def decide_action(self, perception: AntPerception) -> AntAction:
        """
        Machine à état principale
        """
        ant_id = perception.ant_id
        self._update_position(perception)
        self._remember_exploration_state(perception)
        self._update_known_map(perception)

        state = self._state(ant_id)
        action = action = AntAction.MOVE_FORWARD

        if perception.has_food:
            action = self._run_returning(perception)
        elif state == STATE_COLLECTING:
            action = self._run_collecting(perception)
        elif perception.can_see_food():
            self._set_state(ant_id, STATE_COLLECTING)
            action = self._run_collecting(perception)
        elif state == STATE_FOLLOW_FOOD_TO_FOOD:
            action = self._run_follow_food_to_food(perception)
        elif state == STATE_INVERSE_CARRIER:
            action = self._run_inverse_carrier(perception)
        elif state == STATE_EXPLORE_FOOD:
            action = self._run_explore_food(perception)
        elif self._is_escaping_empty_food(perception):
            action = self._run_empty_food_escape(perception)
        elif perception.nearby_ants and state in {STATE_EXPLORING, STATE_FOLLOW_WALL, STATE_GATEWAY_ANALYSIS, STATE_INIT}:
            if self._switch_to_inverse_carrier_if_visible(perception):
                action = self._run_inverse_carrier(perception)
            elif state == STATE_GATEWAY_ANALYSIS:
                action = self._run_gateway_analysis(perception)
            elif state == STATE_INIT:
                action = self._run_init(perception)
            else:
                action = self._run_explo_mode(perception)
        elif state == STATE_GATEWAY_ANALYSIS:
            action = self._run_gateway_analysis(perception)
        elif state == STATE_INIT:
            action = self._run_init(perception)
        elif state in {STATE_EXPLORING, STATE_FOLLOW_WALL}:
            action = self._run_explo_mode(perception)
        else:
            self._set_state(ant_id, STATE_EXPLORING)
            action = self._run_exploring(perception)
        return self._remember_action(perception, action)

    def _run_explo_mode(self, perception):
        """
        Décide de la routine d'exploration
        """
        state=self._state(perception.ant_id)
        if state==STATE_FOLLOW_WALL:
            return self._run_follow_wall(perception)
        elif state==STATE_EXPLORING:
            return self._run_exploring(perception)

    def _run_returning(self, perception):
        """
        Routine de retour à la colonie, avec détection de blocage et stagnation
        """
        ant_id = perception.ant_id
        self.update_food_colony_paths(perception)

        if current_terrain(perception) == TerrainType.COLONY:
            self._set_substate(ant_id, None)
            self._set_state(ant_id, STATE_FOLLOW_FOOD_TO_FOOD)
            if ant_id is not None:
                self._promote_successful_return_path(ant_id)
                self._clear_return_reset_state(ant_id)
                self.outbound_paths[ant_id] = [self.positions.get(ant_id, (0, 0))]
            return AntAction.DROP_FOOD
        reset_return_substate = self._maybe_reset_return_substate(perception)
        if perception.can_see_colony():
            return selectmove(perception, perception.get_colony_direction())
        if reset_return_substate:
            return selectmove(perception, self._home_direction(perception))
        if self._substate(ant_id) == RETURN_SUBSTATE_BLOCKED :
            return selectmove(perception, self._return_trouble_direction(perception))
        if is_blocked(perception) :
            self._set_substate(ant_id, RETURN_SUBSTATE_BLOCKED)
            return selectmove(perception, self._return_trouble_direction(perception))
        if self._substate(ant_id) == RETURN_SUBSTATE_STAGNATING:
            return selectmove(perception, self._return_trouble_direction(perception))
        if self.is_stagnating(perception):
            self._set_substate(ant_id, RETURN_SUBSTATE_STAGNATING)
            return selectmove(perception, self._return_trouble_direction(perception))
        if self._substate(ant_id) == RETURN_SUBSTATE_FOLLOW_PATH:
            direction = (
                self.follow_food_to_colony(perception)
                or self.merge_to_old_foodpath(perception)
                or self._home_direction(perception)
            )
            return selectmove(perception, direction)
        return selectmove(perception, self._home_direction(perception))

    def _run_collecting(self, perception):
        """
        Routine de collect de nourriture
        """
        ant_id = perception.ant_id
        self.update_colony_food_paths(perception)
        if current_terrain(perception) == TerrainType.FOOD:
            self.carrier_directions.pop(ant_id, None)
            self.inverse_carrier_steps.pop(ant_id, None)
            self._set_state(ant_id, STATE_RETURNING)
            self._remember_food(perception)
            return AntAction.PICK_UP_FOOD
        return selectmove(perception, perception.get_food_direction())

    def _run_follow_food_to_food(self, perception):
        """
        Routine de suivi de chemin connu pour retourner à la nourriture
        """
        if self._remembered_food_is_empty(perception):
            self._start_explore_food(perception)
            return self._run_explore_food(perception)

        current = self.positions.get(perception.ant_id, (0, 0))
        direction = (
            self.stable_colony_food_paths.get(perception.ant_id, {}).get(current)
            or self.colony_food_paths.get(perception.ant_id, {}).get(current)
        )
        if direction is None or is_blocked(perception, direction):
            direction = self._known_food_direction(perception)
        if direction is None:
            self._set_state(perception.ant_id, STATE_EXPLORING)
            return self._run_exploring(perception)
        return selectmove(perception, direction)

    def _run_inverse_carrier(self, perception):
        """
        Routine de suivi inverse de porteur 
        """
        ant_id = perception.ant_id
        if perception.can_see_food():
            self._set_state(ant_id, STATE_COLLECTING)
            self.carrier_directions.pop(ant_id, None)
            self.inverse_carrier_steps.pop(ant_id, None)
            return self._run_collecting(perception)

        direction = self.carrier_directions.get(ant_id)
        self._decay_carrier_cone(ant_id)
        if direction is None:
            self._set_state(ant_id, STATE_EXPLORING)
            self.inverse_carrier_steps.pop(ant_id, None)
            return self._exploration_move(perception)
        if self._direction_points_into_avoid_zone(perception, direction):
            self.carrier_directions.pop(ant_id, None)
            self.inverse_carrier_steps.pop(ant_id, None)
            self._set_state(ant_id, STATE_EXPLORING)
            return self._run_empty_food_escape(perception) if self._is_escaping_empty_food(perception) else self._exploration_move(perception)

        self.inverse_carrier_steps[ant_id] = self.inverse_carrier_steps.get(ant_id, 0) + 1
        if self.inverse_carrier_steps[ant_id] > INVERSE_CARRIER_SEARCH_STEPS or self.is_stagnating(perception):
            self.carrier_directions.pop(ant_id, None)
            self.inverse_carrier_steps.pop(ant_id, None)
            self._start_explore_food(perception, target=self._suspected_food_target(perception, direction))
            return self._run_explore_food(perception)

        if is_blocked(perception, direction):
            self.carrier_directions.pop(ant_id, None)
            self.inverse_carrier_steps.pop(ant_id, None)
            self._set_state(ant_id, STATE_FOLLOW_WALL)
            return selectmove(perception, self._best_unstuck_direction(perception))

        self._set_state(ant_id, STATE_INVERSE_CARRIER)
        return self._exploration_move(perception, target=direction)

    def _run_explore_food(self, perception):
        """
        Routine d'exploration de la nourriture quand ancien spot vide
        """
        ant_id = perception.ant_id
        if perception.can_see_food():
            self.food_explore_steps.pop(ant_id, None)
            self.food_search_targets.pop(ant_id, None)
            self.empty_food_escape_steps.pop(ant_id, None)
            self.empty_food_escape_targets.pop(ant_id, None)
            self._set_state(ant_id, STATE_COLLECTING)
            return self._run_collecting(perception)

        steps = self.food_explore_steps.get(ant_id, 0) + 1
        self.food_explore_steps[ant_id] = steps
        if steps > EXPLORE_FOOD_STEPS or self._is_food_search_stagnating(perception):
            current = self.positions.get(ant_id, (0, 0))
            target = self._active_food_search_target(perception)
            self._add_zone(self.avoid_zones, ant_id, target, radius=8)
            self._add_zone(self.avoid_zones, ant_id, current, radius=8)
            self._discard_colony_food_route(ant_id)
            self.outbound_paths.pop(ant_id, None)
            self.food_explore_steps.pop(ant_id, None)
            self.food_search_targets.pop(ant_id, None)
            self._start_empty_food_escape(ant_id, target)
            self._set_state(ant_id, STATE_EXPLORING)
            return selectmove(perception, self._leave_empty_food_area_direction(perception, target))

        direction = self._explore_food_direction(perception)
        return selectmove(perception, direction)

    def _run_gateway_analysis(self, perception):
        """
        routine d'analyse de gateway
        """
        ant_id = perception.ant_id
        steps = self.gateway_analysis_steps.get(ant_id, 0) + 1
        self.gateway_analysis_steps[ant_id] = steps

        if steps > GATEWAY_ANALYSIS_STEPS or not can_see_wall(perception):
            self.gateway_analysis_steps.pop(ant_id, None)
            self._set_state(ant_id, STATE_EXPLORING)
            return selectmove(perception, self._gateway_analysis_direction(perception))

        return selectmove(perception, self._gateway_analysis_direction(perception))

    def _run_exploring(self, perception):
        """
        Routine d'exploration classique, avec détection de blocage et stagnation, et switch vers follow_wall ou gateway_analysis en conséquence
        """
        exploration_mode = self._exploration_mode(perception)
        if is_blocked(perception):
            self._set_state(
                perception.ant_id,
                STATE_FOLLOW_WALL if exploration_mode == EXPLORATION_FAR else STATE_EXPLORING,
            )
            return self._exploration_move(perception)
        if self.is_stagnating(perception):
            self._mark_current_area_to_avoid(perception)
            use_gateway = exploration_mode == EXPLORATION_FAR and can_see_wall(perception)
            self._set_state(
                perception.ant_id,
                STATE_GATEWAY_ANALYSIS if use_gateway else STATE_EXPLORING,
            )
            direction = self._gateway_analysis_direction(perception) if use_gateway else self._best_unstuck_direction(perception)
            return selectmove(perception, direction)
        return self._exploration_move(perception)

    def _run_follow_wall(self, perception):
        """
        Routine de suivi de mur, explo
        """
        if not can_see_wall(perception):
            self._set_state(perception.ant_id, STATE_GATEWAY_ANALYSIS)
            return selectmove(perception, self._gateway_analysis_direction(perception))
        leave_wall_chance = 0.7 if self._exploration_mode(perception) == EXPLORATION_LOCAL else 0.4
        if random.random() < leave_wall_chance:
            self._set_state(perception.ant_id, STATE_EXPLORING)
            return self._exploration_move(perception)

        directions = open_directions(perception)
        if not directions:
            return random_turn()
        direction = max(directions, key=lambda direction: self._wall_follow_score(perception, direction))
        if self._direction_repeats_recent_position(perception, direction):
            direction = self._best_unstuck_direction(perception)
        return selectmove(perception, direction)

    def _run_init(self, perception):
        """
        Va tout droit avec sidestep
        """
        ant_id = perception.ant_id
        if ant_id not in self.seen_ants:
            self.seen_ants.add(ant_id)
            self._plan_initial_sidestep(perception)

        sidestep = self._initial_sidestep_action(perception)
        if sidestep is not None:
            return sidestep
        if is_blocked(perception):
            self._set_state(ant_id, STATE_EXPLORING)
            return random_turn()
        return AntAction.MOVE_FORWARD

    def follow_food_to_colony(self, perception):
        """
        Retourne la direction à suivre pour revenir à la bouffe
        """
        current = self.positions.get(perception.ant_id, (0, 0))
        return (
            self.stable_food_colony_paths.get(perception.ant_id, {}).get(current)
            or self.food_colony_paths.get(perception.ant_id, {}).get(current)
        )

    def is_stagnating(self, perception):
        """
        Détecte la stagnation
        """
        recent = self.recent_positions.get(perception.ant_id, [])
        if len(recent) < RECENT_WINDOW:
            return False

        center_x = sum(x for x, _ in recent) / len(recent)
        center_y = sum(y for _, y in recent) / len(recent)
        spread = max(math.hypot(x - center_x, y - center_y) for x, y in recent)
        return spread <= STAGNATION_RADIUS and len(set(recent)) <= STAGNATION_UNIQUE_LIMIT

    def merge_to_old_foodpath(self, perception):
        """
        Retour à un chemin connu
        """
        current = self.positions.get(perception.ant_id, (0, 0))
        direction_map = (
            self.stable_food_colony_paths.get(perception.ant_id)
            or self.food_colony_paths.get(perception.ant_id, {})
        )
        if current in direction_map:
            self._set_substate(perception.ant_id, RETURN_SUBSTATE_FOLLOW_PATH)
            return direction_map[current]
        direction = self._direct_direction_to_nearest_path_point(perception, direction_map)
        if direction is not None:
            self._set_substate(perception.ant_id, RETURN_SUBSTATE_FOLLOW_PATH)
        return direction

    def update_food_colony_paths(self, perception):
        """
        Met à jour le chemin food -> colony
        """
        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        previous = self.return_paths.setdefault(ant_id, [])
        if not previous or previous[-1] != current:
            previous.append(current)

        self.food_colony_paths.setdefault(ant_id, {}).update(self._path_to_direction_map(previous))

    def update_colony_food_paths(self, perception):
        """
        Met à jour le chemin colonie -> food
        """
        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        path = self.outbound_paths.setdefault(ant_id, [(0, 0)])
        if not path or path[-1] != current:
            path.append(current)
        if len(path) > MAX_PATH_LENGTH:
            del path[: len(path) - MAX_PATH_LENGTH]
        self.colony_food_paths[ant_id] = self._path_to_direction_map(path)

    def _set_colony_food_path(self, ant_id, path):
        if ant_id is None or not path or len(path) < 2:
            return
        direction_map = self._path_to_direction_map(path)
        self.colony_food_paths[ant_id] = direction_map
        self.stable_colony_food_paths[ant_id] = dict(direction_map)

    def _return_trouble_direction(self, perception):
        """
        Direction de secours pour une porteuse bloquee ou stagnante.
        """
        directions = open_directions(perception)
        merge_direction = self.merge_to_old_foodpath(perception)
        wall_escape_direction = self._away_from_nearest_wall_direction(perception)
        home_direction = self._home_direction(perception)
        if not directions:
            return merge_direction or wall_escape_direction or home_direction

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))

        def score(direction):
            dx, dy = Direction.get_delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            value = visible_free_distance(perception, direction) * 0.25
            value -= self._recent_revisit_penalty(ant_id, next_pos) * 1.5
            value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 20) * 0.35
            if merge_direction is not None:
                value -= angular_distance(direction, merge_direction) * 0.8
            if wall_escape_direction is not None:
                value -= angular_distance(direction, wall_escape_direction) * 0.7
            if home_direction is not None:
                value -= angular_distance(direction, home_direction) * 0.25
            return value + random.random() * 0.15

        return max(directions, key=score)

    def _away_from_nearest_wall_direction(self, perception):
        """
        Direction opposee au mur visible le plus proche.
        """
        nearest_wall = None
        nearest_distance = float("inf")
        for (dx, dy), terrain in perception.visible_cells.items():
            if terrain != TerrainType.WALL or (dx == 0 and dy == 0):
                continue
            distance = math.hypot(dx, dy)
            if distance < nearest_distance:
                nearest_wall = (dx, dy)
                nearest_distance = distance
        if nearest_wall is None:
            return None
        return direction_from_vector(-nearest_wall[0], -nearest_wall[1])

    def _maybe_reset_return_substate(self, perception):
        """
        Réinitialise le substate de retour si la fourmi semble bloquée ou stagnante depuis trop longtemps, ou si elle a traversée une gateway
        """
        ant_id = perception.ant_id
        if ant_id is None:
            return False

        self.return_reset_steps[ant_id] = self.return_reset_steps.get(ant_id, 0) + 1
        should_reset = False
        if self.return_reset_steps[ant_id] >= RETURN_SUBSTATE_RESET_STEPS:
            self.return_reset_steps[ant_id] = 0
            should_reset = True

        gateway = self._current_gateway(perception)
        if gateway is None:
            self.return_last_gateway.pop(ant_id, None)
        elif self.return_last_gateway.get(ant_id) != gateway:
            self.return_last_gateway[ant_id] = gateway
            should_reset = True

        if should_reset and self._substate(ant_id) in RETURN_TROUBLE_SUBSTATES:
            self._set_substate(ant_id, None)
            return True
        return False

    def _current_gateway(self, perception):
        """
        Check si fourmi dans gateway
        """
        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        gateways = self.gateways.get(ant_id, set())
        if not gateways:
            return None
        nearby = [
            gateway for gateway in gateways
            if math.hypot(gateway[0] - current[0], gateway[1] - current[1]) <= RETURN_GATEWAY_RESET_RADIUS
        ]
        if not nearby:
            return None
        return min(nearby, key=lambda gateway: math.hypot(gateway[0] - current[0], gateway[1] - current[1]))

    def _clear_return_reset_state(self, ant_id):
        """
        Oublie l'état de réinitialisation pour la fourmi
        """
        self.return_reset_steps.pop(ant_id, None)

    def inverse_direction(self, direction):
        """
        Helper qui retourne la direction inverse
        """
        direction = as_direction(direction)
        if direction is None:
            return None
        return Direction((direction.value + 4) % 8)

    def _update_position(self, perception):
        """
        Met à jour la position de la fourmi
        """
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
        dx, dy = Direction.get_delta(direction)
        new_pos = (x + dx, y + dy)
        self.positions[ant_id] = new_pos
        if not perception.has_food:
            path = self.outbound_paths.setdefault(ant_id, [(0, 0)])
            if not path or path[-1] != new_pos:
                path.append(new_pos)

    def _remember_action(self, perception, action):
        """
        Buffer se souvenant de l'action réalisé
        """
        ant_id = perception.ant_id
        if ant_id is not None:
            expected_move = action == AntAction.MOVE_FORWARD and not is_blocked(perception)
            self.last_actions[ant_id] = (action, perception.direction, expected_move)
        return action

    def _remember_food(self, perception):
        """
        Se souvient d'où elle a trouvé de la food
        """
        ant_id = perception.ant_id
        if ant_id is None:
            return
        position = self.positions.get(ant_id, (0, 0))
        self.food_memory[ant_id] = position

        path_to_food = list(self.outbound_paths.get(ant_id, [(0, 0)]))
        if not path_to_food or path_to_food[-1] != position:
            path_to_food.append(position)
        self._set_colony_food_path(ant_id, path_to_food)

        path_to_colony = list(reversed(path_to_food))
        self.food_colony_paths[ant_id] = self._path_to_direction_map(path_to_colony)
        self.stable_food_colony_paths[ant_id] = dict(self.food_colony_paths[ant_id])
        self.return_paths[ant_id] = [position]

    def _discard_colony_food_route(self, ant_id):
        target = self.food_memory.get(ant_id)
        self.food_memory.pop(ant_id, None)
        self.colony_food_paths.pop(ant_id, None)
        self.stable_colony_food_paths.pop(ant_id, None)

        if target is not None:
            trail = self.carrier_trails.get(ant_id)
            if trail:
                tx, ty = target
                self.carrier_trails[ant_id] = {
                    pos: strength
                    for pos, strength in trail.items()
                    if math.hypot(pos[0] - tx, pos[1] - ty) > 6
                }
            cone = self.carrier_cones.get(ant_id)
            if cone:
                tx, ty = target
                self.carrier_cones[ant_id] = {
                    pos: value
                    for pos, value in cone.items()
                    if math.hypot(pos[0] - tx, pos[1] - ty) > 8
                }

    def _remembered_food_is_empty(self, perception):
        """
        Détecte si le spot de nourriture mémorisé est vide
        """
        ant_id = perception.ant_id
        target = self.food_memory.get(ant_id)
        if target is None:
            return False
        current = self.positions.get(ant_id, (0, 0))
        offset = (target[0] - current[0], target[1] - current[1])
        terrain = perception.visible_cells.get(offset)
        return terrain is not None and terrain != TerrainType.FOOD

    def _forget_empty_food_target(self, perception):
        """
        Oublie la cible de nourriture si EXPLORE_FOOD fail
        """
        ant_id = perception.ant_id
        if ant_id is None:
            return
        target = self.food_memory.get(ant_id, self.positions.get(ant_id, (0, 0)))
        self._add_zone(self.avoid_zones, ant_id, target, radius=8)
        self.food_explore_steps.pop(ant_id, None)
        self.food_search_targets.pop(ant_id, None)
        self._discard_colony_food_route(ant_id)
        self._start_empty_food_escape(ant_id, target)
        self._set_state(ant_id, STATE_EXPLORING)

    def _start_explore_food(self, perception, target=None):
        """
        Lance une courte recherche autour d'un ancien spot vide avant de l'oublier.
        """
        ant_id = perception.ant_id
        if ant_id is None:
            return
        target = target or self.food_memory.get(ant_id)
        self.food_search_targets[ant_id] = target or self.positions.get(ant_id, (0, 0))
        self.food_explore_steps[ant_id] = 0
        self._set_state(ant_id, STATE_EXPLORE_FOOD)

    def _active_food_search_target(self, perception):
        ant_id = perception.ant_id
        return (
            self.food_search_targets.get(ant_id)
            or self.food_memory.get(ant_id)
            or self.positions.get(ant_id, (0, 0))
        )

    def _suspected_food_target(self, perception, direction):
        current = self.positions.get(perception.ant_id, (0, 0))
        direction = as_direction(direction)
        if direction is None:
            return current
        dx, dy = Direction.get_delta(direction)
        return (current[0] + dx * EXPLORE_FOOD_STEPS, current[1] + dy * EXPLORE_FOOD_STEPS)

    def _start_empty_food_escape(self, ant_id, target):
        if ant_id is None or target is None:
            return
        self.empty_food_escape_targets[ant_id] = target
        self.empty_food_escape_steps[ant_id] = 0

    def _is_escaping_empty_food(self, perception):
        ant_id = perception.ant_id
        if ant_id not in self.empty_food_escape_targets:
            return False
        target = self.empty_food_escape_targets[ant_id]
        current = self.positions.get(ant_id, (0, 0))
        steps = self.empty_food_escape_steps.get(ant_id, 0)
        if (
            steps >= EMPTY_FOOD_ESCAPE_STEPS
            or math.hypot(current[0] - target[0], current[1] - target[1]) > EMPTY_FOOD_ESCAPE_RADIUS
        ):
            self.empty_food_escape_targets.pop(ant_id, None)
            self.empty_food_escape_steps.pop(ant_id, None)
            return False
        return True

    def _run_empty_food_escape(self, perception):
        ant_id = perception.ant_id
        if perception.can_see_food():
            self.empty_food_escape_targets.pop(ant_id, None)
            self.empty_food_escape_steps.pop(ant_id, None)
            self._set_state(ant_id, STATE_COLLECTING)
            return self._run_collecting(perception)

        self.empty_food_escape_steps[ant_id] = self.empty_food_escape_steps.get(ant_id, 0) + 1
        target = self.empty_food_escape_targets.get(ant_id)
        return selectmove(perception, self._leave_empty_food_area_direction(perception, target))

    def _is_food_search_stagnating(self, perception):
        """
        Detection plus reactive que la stagnation globale pendant EXPLORE_FOOD.
        """
        ant_id = perception.ant_id
        if self.food_explore_steps.get(ant_id, 0) < 4:
            return False
        recent = self.recent_positions.get(ant_id, [])[-16:]
        if len(recent) < 10:
            return False
        center_x = sum(x for x, _ in recent) / len(recent)
        center_y = sum(y for _, y in recent) / len(recent)
        spread = max(math.hypot(x - center_x, y - center_y) for x, y in recent)
        return spread <= 4 and len(set(recent)) <= 7

    def _leave_empty_food_area_direction(self, perception, target):
        """
        Choisit une direction ouverte qui sort du spot de nourriture vide.
        """
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        target = target or current
        home_direction = self._home_direction(perception)

        def score(direction):
            dx, dy = Direction.get_delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            distance_from_target = math.hypot(next_pos[0] - target[0], next_pos[1] - target[1])
            value = distance_from_target * 1.2
            value += visible_free_distance(perception, direction) * 0.35
            value += self._map_frontier_score(ant_id, direction) * 0.6
            value -= self._recent_revisit_penalty(ant_id, next_pos) * 2.0
            value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 20) * 0.45
            if home_direction is not None:
                value -= angular_distance(direction, home_direction) * 0.15
            if self._inside_zone(self.avoid_zones, ant_id, next_pos):
                value -= 4.0
            if distance_from_target <= 3:
                value -= 3.0
            return value + random.random() * 0.2

        return max(directions, key=score)

    def _promote_successful_return_path(self, ant_id):
        """
        Promeut le chemin de retour food colonie pour les prochains retours
        """
        path = self.return_paths.get(ant_id)
        if not path or len(path) < 2:
            return
        direction_map = self._path_to_direction_map(path)
        self.stable_food_colony_paths[ant_id] = direction_map
        self.food_colony_paths.setdefault(ant_id, {}).update(direction_map)
        self._set_colony_food_path(ant_id, list(reversed(path)))

    def _remember_exploration_state(self, perception):
        """
        Se souvient des cellules visitées et vue
        """
        ant_id = perception.ant_id
        if ant_id is None:
            return

        position = self.positions.get(ant_id, (0, 0))
        visits = self.visit_counts.setdefault(ant_id, {})
        visits[position] = visits.get(position, 0) + 1
        self._remember_seen_empty_cells(perception, position, visits)
        if len(visits) > MAX_VISITED_CELLS:
            self._trim_map_around(visits, position, MAX_VISITED_CELLS)

        recent = self.recent_positions.setdefault(ant_id, [])
        recent.append(position)
        if len(recent) > RECENT_WINDOW:
            del recent[: len(recent) - RECENT_WINDOW]

    def _remember_seen_empty_cells(self, perception, current, visits):
        """
        Se souvient des cellules vides vues
        """
        ant_x, ant_y = current
        for (dx, dy), terrain in perception.visible_cells.items():
            if (dx, dy) == (0, 0) or terrain != TerrainType.EMPTY:
                continue
            position = (ant_x + dx, ant_y + dy)
            visits[position] = visits.get(position, 0) + SEEN_EMPTY_VISIT_WEIGHT

    def _update_known_map(self, perception):
        """
        Met à jour la carte
        """
        ant_id = perception.ant_id
        if ant_id is None:
            return

        ant_x, ant_y = self.positions.get(ant_id, (0, 0))
        known_map = self.known_maps.setdefault(ant_id, {})
        for (dx, dy), terrain in perception.visible_cells.items():
            known_map[(ant_x + dx, ant_y + dy)] = terrain
        if len(known_map) > MAX_KNOWN_CELLS:
            self._trim_map_around(known_map, (ant_x, ant_y), MAX_KNOWN_CELLS)

        if can_see_wall(perception):
            gateways = self.gateways.setdefault(ant_id, set())
            for (dx, dy), terrain in perception.visible_cells.items():
                if terrain != TerrainType.WALL:
                    continue
                for direction in Direction:
                    sx, sy = Direction.get_delta(direction)
                    candidate = (ant_x + dx + sx, ant_y + dy + sy)
                    if known_map.get(candidate) not in (None, TerrainType.WALL):
                        gateways.add(candidate)
            if len(gateways) > MAX_GATEWAYS:
                self.gateways[ant_id] = set(
                    sorted(gateways, key=lambda pos: math.hypot(pos[0] - ant_x, pos[1] - ant_y))[:MAX_GATEWAYS]
                )

    def _path_to_direction_map(self, path):
        directions = {}
        for current, next_pos in zip(path, path[1:]):
            directions[current] = direction_from_vector(next_pos[0] - current[0], next_pos[1] - current[1])
        return directions

    def _home_direction(self, perception):
        x, y = self.positions.get(perception.ant_id, (0, 0))
        if x == 0 and y == 0:
            return None
        return direction_from_vector(-x, -y)

    def _known_food_direction(self, perception):
        ant_id = perception.ant_id
        target = self.food_memory.get(ant_id)
        mapped = self._mapped_direction_to(ant_id, target)
        if mapped is not None and not is_blocked(perception, mapped):
            return mapped

        current = self.positions.get(ant_id, (0, 0))
        direct = direction_from_vector(target[0] - current[0], target[1] - current[1]) if target is not None else None
        if direct is not None and not is_blocked(perception, direct):
            return direct

        directions = open_directions(perception)
        if not directions or target is None:
            return None

        def distance_after_step(direction):
            dx, dy = Direction.get_delta(direction)
            return math.hypot(target[0] - current[0] - dx, target[1] - current[1] - dy)

        return min(directions, key=distance_after_step)

    def _sector_direction(self, perception):
        sectors = [
            (1, -1), (-1, -1), (1, 1), (-1, 1),
            (1, 0), (0, 1), (-1, 0), (0, -1),
        ]
        return DIRECTION_BY_STEP[sectors[(perception.ant_id or 0) % len(sectors)]]

    def _exploration_mode(self, perception):
        offset = (perception.ant_id or 0) * (EXPLORATION_MODE_PERIOD // 3)
        phase = ((perception.steps_taken + offset) // EXPLORATION_MODE_PERIOD) % 2
        return EXPLORATION_FAR if phase else EXPLORATION_LOCAL

    def _exploration_direction(self, perception):
        if self._exploration_mode(perception) == EXPLORATION_LOCAL:
            return self._sector_direction(perception)
        away_from_colony = self.inverse_direction(self._home_direction(perception))
        return away_from_colony or self._sector_direction(perception)

    def _exploration_move(self, perception, target=None):
        return selectmove(
            perception,
            self._best_open_direction(
                perception,
                target=target or self._exploration_direction(perception),
                force_exploration=True,
            ),
        )

    def _best_open_direction(self, perception, target=None, force_exploration=False):
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))

        def score(direction):
            dx, dy = Direction.get_delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            value = random.random() * 0.15
            value += visible_free_distance(perception, direction) * 0.08
            value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 8) * 0.25
            value -= self._recent_revisit_penalty(ant_id, next_pos)
            value += self._map_frontier_score(ant_id, direction)
            value += self._carrier_cone_score(ant_id, direction) * CARRIER_CONE_WEIGHT
            value += self._carrier_trail_score(ant_id, direction) * CARRIER_TRAIL_EXPLORATION_WEIGHT
            if direction == perception.direction:
                value += 0.35
            if target is not None:
                value -= angular_distance(direction, target) * 0.55
            away_from_colony = self.inverse_direction(self._home_direction(perception))
            exploration_mode = self._exploration_mode(perception)
            if force_exploration and exploration_mode == EXPLORATION_FAR and away_from_colony is not None:
                value -= angular_distance(direction, away_from_colony) * EXPLORATION_AWAY_FROM_COLONY_WEIGHT
            if force_exploration and exploration_mode == EXPLORATION_LOCAL:
                value -= self._wall_contact_score(perception, direction) * LOCAL_WALL_CONTACT_PENALTY
            if self._inside_zone(self.avoid_zones, ant_id, next_pos):
                value -= 3.0 if force_exploration else 1.2
            return value

        return max(directions, key=score)

    def _best_unstuck_direction(self, perception):
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))

        def score(direction):
            dx, dy = Direction.get_delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            value = visible_free_distance(perception, direction) * 0.2
            value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 20) * 0.5
            value -= self._recent_revisit_penalty(ant_id, next_pos) * 1.5
            if self._inside_zone(self.avoid_zones, ant_id, next_pos):
                value -= 4.0
            return value + random.random() * 0.25

        return max(directions, key=score)

    def _map_frontier_score(self, ant_id, direction):
        known_map = self.known_maps.get(ant_id, {})
        current = self.positions.get(ant_id, (0, 0))
        dx, dy = Direction.get_delta(direction)
        score = 0.0
        for step in range(1, FRONTIER_LOOKAHEAD + 1):
            position = (current[0] + dx * step, current[1] + dy * step)
            terrain = known_map.get(position)
            if terrain == TerrainType.WALL:
                score -= 1.5 / step
                break
            if terrain is None:
                score += 1.4 / step
        return score

    def _food_direction_from_carrier(self, perception, carrier_offset):
        ant_x, ant_y = self.positions.get(perception.ant_id, (0, 0))
        carrier_x = ant_x + carrier_offset[0]
        carrier_y = ant_y + carrier_offset[1]
        return direction_from_vector(carrier_x, carrier_y)

    def _start_inverse_carrier(self, perception, direction):
        ant_id = perception.ant_id
        if ant_id is None or direction is None:
            return
        self.carrier_directions[ant_id] = direction
        self.inverse_carrier_steps[ant_id] = 0
        self._mark_carrier_cone(perception, direction)

    def _carrier_direction_from_visible_ant(self, perception):
        if perception.can_see_food():
            return None

        carriers = [offset for offset, has_food in perception.nearby_ants if has_food]
        if not carriers:
            return None

        direction = self._food_direction_from_carrier(
            perception,
            min(carriers, key=lambda offset: math.hypot(*offset)),
        )
        self._remember_carrier_trail(perception, direction)
        return direction

    def _switch_to_inverse_carrier_if_visible(self, perception):
        ant_id = perception.ant_id
        if ant_id is None:
            return False

        carriers = [offset for offset, has_food in perception.nearby_ants if has_food]
        if not carriers:
            return False

        carrier_offset = min(carriers, key=lambda offset: math.hypot(*offset))
        carrier_direction = self._food_direction_from_carrier(perception, carrier_offset)
        if carrier_direction is None:
            return False
        if self._direction_points_into_avoid_zone(perception, carrier_direction):
            return False

        self._remember_carrier_trail(perception, carrier_direction)
        self._start_inverse_carrier(perception, carrier_direction)
        self._set_state(ant_id, STATE_INVERSE_CARRIER)
        return True

    def _direction_points_into_avoid_zone(self, perception, direction, max_steps=TRAIL_LENGTH):
        ant_id = perception.ant_id
        direction = as_direction(direction)
        if ant_id is None or direction is None:
            return False
        current = self.positions.get(ant_id, (0, 0))
        dx, dy = Direction.get_delta(direction)
        for step in range(1, max_steps + 1):
            position = (current[0] + dx * step, current[1] + dy * step)
            if self._inside_zone(self.avoid_zones, ant_id, position):
                return True
        return False

    def _remember_carrier_trail(self, perception, direction):
        ant_id = perception.ant_id
        if ant_id is None or direction is None:
            return
        if self._direction_points_into_avoid_zone(perception, direction):
            return

        current = self.positions.get(ant_id, (0, 0))
        known_map = self.known_maps.get(ant_id, {})
        trail = self.carrier_trails.setdefault(ant_id, {})
        dx, dy = Direction.get_delta(direction)

        for step in range(1, TRAIL_LENGTH + 1):
            position = (current[0] + dx * step, current[1] + dy * step)
            if self._inside_zone(self.avoid_zones, ant_id, position):
                break
            if position not in known_map:
                strength = 1.2 - step / (TRAIL_LENGTH + 5)
                trail[position] = max(trail.get(position, 0.0), strength)

        if len(trail) > MAX_TRAIL_CELLS:
            self.carrier_trails[ant_id] = dict(
                sorted(trail.items(), key=lambda item: item[1], reverse=True)[:MAX_TRAIL_CELLS]
            )

    def _carrier_trail_score(self, ant_id, direction):
        trail = self.carrier_trails.get(ant_id)
        if not trail:
            return 0.0

        current = self.positions.get(ant_id, (0, 0))
        known_map = self.known_maps.get(ant_id, {})
        dx, dy = Direction.get_delta(direction)
        score = 0.0
        for step in range(1, FRONTIER_LOOKAHEAD + 6):
            position = (current[0] + dx * step, current[1] + dy * step)
            if self._inside_zone(self.avoid_zones, ant_id, position):
                return -CARRIER_TRAIL_SCORE_WEIGHT
            if position not in known_map:
                score += trail.get(position, 0.0) / step
        return score * CARRIER_TRAIL_SCORE_WEIGHT

    def _carrier_trail_direction(self, perception):
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        best_score, best_direction = max(
            ((self._carrier_trail_score(ant_id, direction), direction) for direction in directions),
            key=lambda item: item[0],
        )
        return best_direction if best_score >= CARRIER_TRAIL_MIN_ACTION_SCORE else None

    def _mark_carrier_cone(self, perception, food_direction):
        ant_id = perception.ant_id
        if ant_id is None or food_direction is None:
            return
        if self._direction_points_into_avoid_zone(perception, food_direction, max_steps=CARRIER_CONE_LENGTH):
            return

        current = self.positions.get(ant_id, (0, 0))
        forward_x, forward_y = Direction.get_delta(food_direction)
        cone = self.carrier_cones.setdefault(ant_id, {})

        for depth in range(1, CARRIER_CONE_LENGTH + 1):
            center = (current[0] + forward_x * depth, current[1] + forward_y * depth)
            width = min(CARRIER_CONE_WIDTH, 1 + depth // 3)
            for side in range(-width, width + 1):
                if forward_x == 0:
                    position = (center[0] + side, center[1])
                elif forward_y == 0:
                    position = (center[0], center[1] + side)
                else:
                    position = (center[0] + side, center[1] - side)
                strength = max(0.2, 1.0 - depth / (CARRIER_CONE_LENGTH + 1))
                old_ttl, old_strength = cone.get(position, (0, 0.0))
                cone[position] = (max(old_ttl, CARRIER_CONE_TTL), max(old_strength, strength))

    def _decay_carrier_cone(self, ant_id):
        cone = self.carrier_cones.get(ant_id)
        if not cone:
            return

        decayed = {}
        for position, (ttl, strength) in cone.items():
            if ttl > 1:
                decayed[position] = (ttl - 1, strength)
        if decayed:
            self.carrier_cones[ant_id] = decayed
        else:
            self.carrier_cones.pop(ant_id, None)

    def _carrier_cone_score(self, ant_id, direction):
        cone = self.carrier_cones.get(ant_id)
        if not cone:
            return 0.0

        current = self.positions.get(ant_id, (0, 0))
        dx, dy = Direction.get_delta(direction)
        score = 0.0
        for step in range(1, FRONTIER_LOOKAHEAD + 3):
            position = (current[0] + dx * step, current[1] + dy * step)
            if self._inside_zone(self.avoid_zones, ant_id, position):
                return -1.0
            _, strength = cone.get(position, (0, 0.0))
            score += strength / step
        return score

    def _wall_follow_score(self, perception, direction):
        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        dx, dy = Direction.get_delta(direction)
        next_pos = (current[0] + dx, current[1] + dy)
        wall_neighbors = 0
        for side in (-1, 1):
            side_direction = Direction((direction.value + side) % 8)
            sx, sy = Direction.get_delta(side_direction)
            if perception.visible_cells.get((dx + sx, dy + sy)) == TerrainType.WALL:
                wall_neighbors += 1
        value = wall_neighbors * 2.0 + visible_free_distance(perception, direction) * 0.2
        value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 20) * 0.55
        value -= self._recent_revisit_penalty(ant_id, next_pos) * 2.0
        if direction == perception.direction:
            value += 0.2
        return value

    def _wall_contact_score(self, perception, direction):
        dx, dy = Direction.get_delta(direction)
        score = 0
        for offset in (-2, -1, 1, 2):
            side_direction = Direction((direction.value + offset) % 8)
            sx, sy = Direction.get_delta(side_direction)
            if perception.visible_cells.get((dx + sx, dy + sy)) == TerrainType.WALL:
                score += 1
        return score

    def _explore_food_direction(self, perception):
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        away_from_colony = self.inverse_direction(self._home_direction(perception))

        def score(direction):
            dx, dy = Direction.get_delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            value = visible_free_distance(perception, direction) * 0.25
            value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 20) * 0.6
            value -= self._recent_revisit_penalty(ant_id, next_pos) * 2.0
            value += self._map_frontier_score(ant_id, direction) * 0.8
            if away_from_colony is not None:
                value -= angular_distance(direction, away_from_colony) * 0.25
            if self._inside_zone(self.avoid_zones, ant_id, next_pos):
                value -= 3.0
            return value + random.random() * 0.2

        return max(directions, key=score)

    def _gateway_analysis_direction(self, perception):
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        gateways = self.gateways.get(ant_id, set())

        def score(direction):
            dx, dy = Direction.get_delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            value = visible_free_distance(perception, direction) * 0.35
            value += random.random() * 0.1

            if next_pos not in self.known_maps.get(ant_id, {}):
                value += 1.2
            value -= min(self.visit_counts.get(ant_id, {}).get(next_pos, 0), 8) * 0.2
            value -= self._recent_revisit_penalty(ant_id, next_pos)

            if gateways:
                nearest_gateway = min(
                    math.hypot(next_pos[0] - gx, next_pos[1] - gy)
                    for gx, gy in gateways
                )
                value -= nearest_gateway * 0.08
            return value

        return max(directions, key=score)

    def _recent_revisit_penalty(self, ant_id, position):
        recent = self.recent_positions.get(ant_id, [])
        penalty = 0.0
        for age, old_position in enumerate(reversed(recent[-12:]), start=1):
            if old_position == position:
                penalty += RECENT_REVISIT_PENALTY * (13 - age)
        return penalty

    def _direction_repeats_recent_position(self, perception, direction):
        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        dx, dy = Direction.get_delta(direction)
        next_pos = (current[0] + dx, current[1] + dy)
        return next_pos in self.recent_positions.get(ant_id, [])[-8:]

    def _direct_direction_to_nearest_path_point(self, perception, direction_map):
        if not direction_map:
            return None

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        path_points = set(direction_map)
        for point, direction in direction_map.items():
            dx, dy = Direction.get_delta(direction)
            path_points.add((point[0] + dx, point[1] + dy))

        targets = sorted(path_points, key=lambda point: math.hypot(point[0] - current[0], point[1] - current[1]))
        fallback = None
        for target in targets:
            if target == current:
                direction = direction_map.get(current)
            else:
                direction = direction_from_vector(target[0] - current[0], target[1] - current[1])
            if direction is None:
                continue
            if fallback is None:
                fallback = direction
            if not is_blocked(perception, direction):
                return direction
        return fallback

    def _mapped_direction_to(self, ant_id, target):
        if ant_id is None or target is None:
            return None
        known_map = self.known_maps.get(ant_id, {})
        current = self.positions.get(ant_id, (0, 0))
        if current == target:
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
                dx, dy = Direction.get_delta(direction)
                neighbor = (position[0] + dx, position[1] + dy)
                if known_map.get(neighbor) == TerrainType.WALL:
                    continue
                if neighbor not in known_map and neighbor != target:
                    continue

                new_cost = current_cost + 1.0 + min(self.visit_counts.get(ant_id, {}).get(neighbor, 0), 8) * 0.12
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
        return direction_from_vector(step[0] - current[0], step[1] - current[1])

    def _mark_current_area_to_avoid(self, perception):
        ant_id = perception.ant_id
        recent = self.recent_positions.get(ant_id, [])
        if recent:
            center = (
                sum(x for x, _ in recent) / len(recent),
                sum(y for _, y in recent) / len(recent),
            )
            self._add_zone(self.avoid_zones, ant_id, center, radius=7)

    def _add_zone(self, zone_store, ant_id, center, radius):
        zones = zone_store.setdefault(ant_id, [])
        zone = (int(round(center[0])), int(round(center[1])), radius)
        if zone not in zones:
            zones.append(zone)
        if len(zones) > 8:
            del zones[: len(zones) - 8]

    def _inside_zone(self, zone_store, ant_id, position):
        if position is None:
            return False
        return any(
            math.hypot(position[0] - x, position[1] - y) <= radius
            for x, y, radius in zone_store.get(ant_id, [])
        )

    def _trim_map_around(self, mapping, center, max_size):
        keep = set(
            sorted(mapping, key=lambda pos: math.hypot(pos[0] - center[0], pos[1] - center[1]))[:max_size]
        )
        for position in list(mapping):
            if position not in keep:
                del mapping[position]

    def _plan_initial_sidestep(self, perception):
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
        plan = self.sidestep_plans.get(perception.ant_id)
        if not plan:
            return None

        action = plan.pop(0)
        if action == AntAction.MOVE_FORWARD and is_blocked(perception):
            action = plan.pop(0) if plan else None
        if not plan:
            self.sidestep_plans.pop(perception.ant_id, None)
        return action
