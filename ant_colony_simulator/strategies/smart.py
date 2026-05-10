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

RECENT_WINDOW = 50
STAGNATION_RADIUS = 8
STAGNATION_UNIQUE_LIMIT = 5
FRONTIER_LOOKAHEAD = 3
EXPLORE_FOOD_STEPS = 10
EMPTY_FOOD_ESCAPE_STEPS = 14
EMPTY_FOOD_ESCAPE_RADIUS = 10
RECENT_REVISIT_PENALTY = 0.65
SEEN_EMPTY_VISIT_WEIGHT = 1
EXPLORATION_AWAY_FROM_COLONY_WEIGHT = 0.85
EXPLORATION_MODE_PERIOD = 80
LOCAL_WALL_CONTACT_PENALTY = 0.9
RETURN_SUBSTATE_RESET_STEPS = 150
RETURN_HOME_PHEROMONE_RESET_STEPS = 25
FOOD_PHEROMONE_THRESHOLD = 8.0
HOME_PHEROMONE_THRESHOLD = 10.0
FOOD_PHEROMONE_FOLLOW_STEPS = 80
FOOD_PHEROMONE_DEPOSIT_INTERVAL = 3
HOME_PHEROMONE_DEPOSIT_INTERVAL = 3
HOME_PHEROMONE_EXPLORED_RADIUS = 4
HOME_PHEROMONE_EXPLORED_WEIGHT = 2.0
SHARED_EXPLORED_LOOKAHEAD = 3
SHARED_EXPLORED_MAX_CELLS = MAX_ENVIRONMENT_CELLS

STATE_INIT = "init"
STATE_EXPLORING = "exploring"
STATE_COLLECTING = "collecting"
STATE_RETURNING = "returning"
STATE_FOLLOW_FOOD_TO_FOOD = "follow_food_to_food"
STATE_EXPLORE_FOOD = "explore_food"
STATE_FOLLOW_WALL = "follow_wall"
STATE_FOLLOW_FOOD_PHEM = "follow_food_phem"

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
ROLE_CARRIER = "carrier"
ROLE_EXPLORER = "explorer"

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


class SmartStrategy(AntStrategy):
    """Role-aware memory strategy based on the non-cooperative state machine."""

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
        self.shared_explored_cells = {}

        self.food_explore_steps = {}
        self.food_search_targets = {}
        self.empty_food_escape_steps = {}
        self.empty_food_escape_targets = {}
        self.food_pheromone_follow_steps = {}
        self.return_reset_steps = {}
        self.return_home_pheromone_steps = {}

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

    def _role(self, ant_id):
        return ROLE_CARRIER if (ant_id or 0) % 10 == 0 else ROLE_EXPLORER

    def decide_action(self, perception: AntPerception) -> AntAction:
        """
        Machine à état principale
        """
        ant_id = perception.ant_id
        self._update_position(perception)
        self._remember_exploration_state(perception)
        self._update_known_map(perception)
        self._remember_home_pheromone_exploration(perception)

        state = self._state(ant_id)
        action = AntAction.MOVE_FORWARD  

        if perception.has_food:
            action = self._run_returning(perception)
        elif state == STATE_COLLECTING:
            action = self._run_collecting(perception)
        elif perception.can_see_food():
            self._set_state(ant_id, STATE_COLLECTING)
            action = self._run_collecting(perception)
        elif state == STATE_FOLLOW_FOOD_TO_FOOD:
            action = self._run_follow_food_to_food(perception)
        elif state == STATE_EXPLORE_FOOD:
            action = self._run_explore_food(perception)
        elif state == STATE_FOLLOW_FOOD_PHEM:
            action = self._run_follow_food_phem(perception)
        elif self._should_start_food_pheromone_follow(perception):
            self._set_state(ant_id, STATE_FOLLOW_FOOD_PHEM)
            action = self._run_follow_food_phem(perception)
        elif self._is_escaping_empty_food(perception):
            action = self._run_empty_food_escape(perception)
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
            if self._role(perception.ant_id) == ROLE_CARRIER:
                return self._run_carrier_exploring(perception)
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
                self.return_home_pheromone_steps.pop(ant_id, None)
                self._clear_pheromone_memory(ant_id, "food")
                self.food_pheromone_follow_steps.pop(ant_id, None)
                self.outbound_paths[ant_id] = [self.positions.get(ant_id, (0, 0))]
            return AntAction.DROP_FOOD

        reset_return_substate = self._maybe_reset_return_substate(perception)
        if perception.can_see_colony():
            return selectmove(perception, perception.get_colony_direction())

        home_direction = self._home_direction(perception)
        if is_blocked(perception) :
            self._set_substate(ant_id, RETURN_SUBSTATE_BLOCKED)
        elif self.is_stagnating(perception):
            self._set_substate(ant_id, RETURN_SUBSTATE_STAGNATING)

        if self._substate(ant_id) in RETURN_TROUBLE_SUBSTATES:
            home_pheromone_action = self._return_home_pheromone_action(perception)
            if home_pheromone_action is not None:
                return home_pheromone_action
            food_pheromone = self._food_pheromone_colony_direction(perception)
            if food_pheromone is not None:
                if self._should_deposit_food_pheromone(perception):
                    return AntAction.DEPOSIT_FOOD_PHEROMONE
                return selectmove(perception, food_pheromone)
            if reset_return_substate:
                return selectmove(perception, home_direction)
            direction = (
                self.follow_food_to_colony(perception)
                or self.merge_to_old_foodpath(perception)
                or home_direction
            )
            return selectmove(perception, direction)

        self.return_home_pheromone_steps.pop(ant_id, None)
        if self._should_deposit_food_pheromone(perception):
            return AntAction.DEPOSIT_FOOD_PHEROMONE
        if reset_return_substate:
            return selectmove(perception, home_direction)

        direction = home_direction or self.follow_food_to_colony(perception) or self.merge_to_old_foodpath(perception)
        return selectmove(perception, direction)

    def _run_collecting(self, perception):
        """
        Routine de collect de nourriture
        """
        ant_id = perception.ant_id
        self.update_colony_food_paths(perception)
        if current_terrain(perception) == TerrainType.FOOD:
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

    def _run_follow_food_phem(self, perception):
        """
        Suit une piste de phéromone food si la fourmi n'a pas encore de spot connu.
        """
        ant_id = perception.ant_id
        if ant_id in self.food_memory:
            self.food_pheromone_follow_steps.pop(ant_id, None)
            self._set_state(ant_id, STATE_FOLLOW_FOOD_TO_FOOD)
            return self._run_follow_food_to_food(perception)
        if perception.can_see_food():
            self.food_pheromone_follow_steps.pop(ant_id, None)
            self._set_state(ant_id, STATE_COLLECTING)
            return self._run_collecting(perception)

        steps = self.food_pheromone_follow_steps.get(ant_id, 0) + 1
        self.food_pheromone_follow_steps[ant_id] = steps
        direction = self._food_pheromone_direction(perception)
        if direction is None or steps > FOOD_PHEROMONE_FOLLOW_STEPS:
            self.food_pheromone_follow_steps.pop(ant_id, None)
            self._set_state(ant_id, STATE_EXPLORING)
            return self._run_exploring(perception)
        return selectmove(perception, direction)

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

    def _run_carrier_exploring(self, perception):
        """
        Exploration role-aware: carriers patrol around the colony instead of following other ants.
        """
        ant_id = perception.ant_id
        if is_blocked(perception):
            self._set_state(ant_id, STATE_EXPLORING)
            return self._exploration_move(perception, target=self._carrier_patrol_direction(perception))
        if self.is_stagnating(perception):
            self._mark_current_area_to_avoid(perception)
            return selectmove(perception, self._best_unstuck_direction(perception))
        deposit = self._exploration_home_pheromone_action(perception)
        if deposit is not None:
            return deposit
        return self._exploration_move(perception, target=self._carrier_patrol_direction(perception))

    def _run_exploring(self, perception):
        """
        Routine d'exploration classique, avec détection de blocage et stagnation.
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
            self._set_state(perception.ant_id, STATE_EXPLORING)
            direction = self._best_unstuck_direction(perception)
            return selectmove(perception, direction)
        deposit = self._exploration_home_pheromone_action(perception)
        if deposit is not None:
            return deposit
        return self._exploration_move(perception)

    def _run_follow_wall(self, perception):
        """
        Routine de suivi de mur, explo
        """
        deposit = self._exploration_home_pheromone_action(perception)
        if deposit is not None:
            return deposit

        leave_wall_chance = 0.15 if self._exploration_mode(perception) == EXPLORATION_LOCAL else 0.0
        if leave_wall_chance and random.random() < leave_wall_chance:
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
        deposit = self._exploration_home_pheromone_action(perception)
        if deposit is not None:
            return deposit
        return AntAction.MOVE_FORWARD
    
    def deposit_foodphem(self, perception):
        """
        Dépose une phéromone de nourriture pour aider les autres fourmis à trouver la nourriture
        """
        if self._should_deposit_food_pheromone(perception):
            return AntAction.DEPOSIT_FOOD_PHEROMONE
        return None
    
    def deposit_colonyphem_routin(self, perception): # nouvelle état
        """
        Dépose une phéromone colonie pour marquer une zone explorée.
        """
        if self._should_deposit_home_pheromone(perception):
            return AntAction.DEPOSIT_HOME_PHEROMONE
        return None

    def _should_deposit_food_pheromone(self, perception):
        if not perception.has_food:
            return False
        ant_id = perception.ant_id or 0
        if perception.steps_taken <= 2:
            return True
        return (perception.steps_taken + ant_id) % FOOD_PHEROMONE_DEPOSIT_INTERVAL == 0

    def _should_deposit_home_pheromone(self, perception):
        if perception.has_food or perception.can_see_colony():
            return False
        if self._state(perception.ant_id) not in {STATE_INIT, STATE_EXPLORING, STATE_FOLLOW_WALL}:
            return False
        return (perception.steps_taken + (perception.ant_id or 0)) % HOME_PHEROMONE_DEPOSIT_INTERVAL == 0

    def _exploration_home_pheromone_action(self, perception):
        if not self._should_deposit_home_pheromone(perception):
            return None
        self._mark_shared_explored_zone(
            self.positions.get(perception.ant_id, (0, 0)),
            radius=HOME_PHEROMONE_EXPLORED_RADIUS,
            amount=HOME_PHEROMONE_EXPLORED_WEIGHT,
        )
        return AntAction.DEPOSIT_HOME_PHEROMONE

    def _should_start_food_pheromone_follow(self, perception):
        if perception.ant_id in self.food_memory:
            return False
        if self._state(perception.ant_id) not in {STATE_INIT, STATE_EXPLORING, STATE_FOLLOW_WALL}:
            return False
        return self._food_pheromone_direction(perception) is not None

    def _food_pheromone_direction(self, perception):
        return self._pheromone_direction(
            perception,
            perception.food_pheromone,
            threshold=FOOD_PHEROMONE_THRESHOLD,
            outward=True,
            remembered_kind="food",
        )

    def _food_pheromone_colony_direction(self, perception):
        return self._pheromone_direction(
            perception,
            perception.food_pheromone,
            threshold=FOOD_PHEROMONE_THRESHOLD,
            outward=False,
            inward=True,
            remembered_kind="food",
        )

    def _home_pheromone_colony_direction(self, perception):
        return self._pheromone_direction(
            perception,
            perception.home_pheromone,
            threshold=HOME_PHEROMONE_THRESHOLD,
            outward=False,
            inward=True,
            remembered_kind="home",
        )

    def _return_home_pheromone_action(self, perception):
        ant_id = perception.ant_id
        steps = self.return_home_pheromone_steps.get(ant_id, 0)
        if steps >= RETURN_HOME_PHEROMONE_RESET_STEPS:
            self.return_home_pheromone_steps.pop(ant_id, None)
            self._set_substate(ant_id, None)
            return None

        direction = self._home_pheromone_colony_direction(perception)
        if direction is None:
            self.return_home_pheromone_steps.pop(ant_id, None)
            return None

        self.return_home_pheromone_steps[ant_id] = steps + 1
        if self._should_deposit_food_pheromone(perception):
            return AntAction.DEPOSIT_FOOD_PHEROMONE
        return selectmove(perception, direction)

    def _pheromone_direction(self, perception, pheromone_map, threshold, outward=False, inward=False, remembered_kind=None):
        directions = open_directions(perception)
        if not directions:
            return None

        ant_id = perception.ant_id
        current = self.positions.get(ant_id, (0, 0))
        current_distance = math.hypot(*current)
        candidates = []

        for direction in directions:
            dx, dy = Direction.get_delta(direction)
            next_pos = (current[0] + dx, current[1] + dy)
            next_distance = math.hypot(*next_pos)
            if outward and next_distance <= current_distance:
                continue
            if inward and next_distance >= current_distance:
                continue

            score = 0.0
            for (pdx, pdy), value in pheromone_map.items():
                if value < threshold or (pdx, pdy) == (0, 0):
                    continue
                absolute = (current[0] + pdx, current[1] + pdy)
                if inward and math.hypot(*absolute) >= current_distance:
                    continue
                if outward and math.hypot(*absolute) <= current_distance:
                    continue
                trail_direction = direction_from_vector(absolute[0] - current[0], absolute[1] - current[1])
                angle = angular_distance(direction, trail_direction)
                if angle > 2:
                    continue
                score += value * ((3 - angle) / 3) / max(1.0, math.hypot(pdx, pdy))

            if score >= threshold:
                if direction == perception.direction:
                    score *= 1.08
                score += visible_free_distance(perception, direction) * 0.3
                candidates.append((score, direction))

        if not candidates:
            return None
        best = max(candidates, key=lambda item: item[0])[0]
        useful = [(score, direction) for score, direction in candidates if score >= best * 0.65]
        return random.choices([direction for _, direction in useful], weights=[score for score, _ in useful], k=1)[0]

    def _remember_home_pheromone_exploration(self, perception):
        ant_id = perception.ant_id
        if ant_id is None:
            return

        current = self.positions.get(ant_id, (0, 0))
        for (dx, dy), value in perception.home_pheromone.items():
            if value < HOME_PHEROMONE_THRESHOLD:
                continue
            self._mark_shared_explored_zone(
                (current[0] + dx, current[1] + dy),
                radius=HOME_PHEROMONE_EXPLORED_RADIUS,
                amount=value / 100.0,
            )

    def _clear_pheromone_memory(self, ant_id, kind):
        return None

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
            value -= self._shared_explored_penalty(next_pos, returning=True)
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
        Réinitialise le substate de retour si la fourmi semble bloquée ou stagnante depuis trop longtemps.
        """
        ant_id = perception.ant_id
        if ant_id is None:
            return False

        self.return_reset_steps[ant_id] = self.return_reset_steps.get(ant_id, 0) + 1
        should_reset = False
        if self.return_reset_steps[ant_id] >= RETURN_SUBSTATE_RESET_STEPS:
            self.return_reset_steps[ant_id] = 0
            should_reset = True

        if should_reset and self._substate(ant_id) in RETURN_TROUBLE_SUBSTATES:
            self._set_substate(ant_id, None)
            return True
        return False

    def _clear_return_reset_state(self, ant_id):
        """
        Oublie l'état de réinitialisation pour la fourmi
        """
        self.return_reset_steps.pop(ant_id, None)
        self.return_home_pheromone_steps.pop(ant_id, None)

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

    def _sector_direction(self, perception, offset=0):
        sectors = [
            (1, -1), (-1, -1), (1, 1), (-1, 1),
            (1, 0), (0, 1), (-1, 0), (0, -1),
        ]
        return DIRECTION_BY_STEP[sectors[((perception.ant_id or 0) + offset) % len(sectors)]]

    def _exploration_mode(self, perception):
        offset = (perception.ant_id or 0) * (EXPLORATION_MODE_PERIOD // 3)
        phase = ((perception.steps_taken + offset) // EXPLORATION_MODE_PERIOD) % 2
        return EXPLORATION_FAR if phase else EXPLORATION_LOCAL

    def _exploration_direction(self, perception):
        if self._exploration_mode(perception) == EXPLORATION_LOCAL:
            return self._sector_direction(perception)
        away_from_colony = self.inverse_direction(self._home_direction(perception))
        return away_from_colony or self._sector_direction(perception)

    def _carrier_patrol_direction(self, perception):
        current = self.positions.get(perception.ant_id, (0, 0))
        distance = math.hypot(*current)
        if distance > 16:
            return self._home_direction(perception)
        if distance < 5:
            return self._sector_direction(perception, offset=2)

        radial = direction_from_vector(current[0], current[1])
        side = 2 if (perception.ant_id or 0) % 2 == 0 else -2
        return Direction((radial.value + side) % 8)

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
            value -= self._shared_explored_direction_penalty(current, direction)
            value += self._map_frontier_score(ant_id, direction)
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
            value -= self._shared_explored_direction_penalty(current, direction)
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
        value -= self._shared_explored_direction_penalty(current, direction)
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

    def _mark_shared_explored_zone(self, center, radius, amount=1.0):
        if center is None:
            return
        cx, cy = int(round(center[0])), int(round(center[1]))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if math.hypot(dx, dy) > radius:
                    continue
                position = (cx + dx, cy + dy)
                self.shared_explored_cells[position] = min(
                    12.0,
                    self.shared_explored_cells.get(position, 0.0) + amount,
                )
        if len(self.shared_explored_cells) > SHARED_EXPLORED_MAX_CELLS:
            keep = set(
                sorted(
                    self.shared_explored_cells,
                    key=lambda pos: math.hypot(pos[0] - cx, pos[1] - cy),
                )[:SHARED_EXPLORED_MAX_CELLS]
            )
            for position in list(self.shared_explored_cells):
                if position not in keep:
                    del self.shared_explored_cells[position]

    def _shared_explored_penalty(self, position, returning=False):
        value = self.shared_explored_cells.get(position, 0.0)
        if value <= 0:
            return 0.0
        return value * (0.25 if returning else 1.35)

    def _shared_explored_direction_penalty(self, current, direction, returning=False):
        dx, dy = Direction.get_delta(direction)
        penalty = 0.0
        for step in range(1, SHARED_EXPLORED_LOOKAHEAD + 1):
            position = (current[0] + dx * step, current[1] + dy * step)
            penalty += self._shared_explored_penalty(position, returning=returning) / step
        return penalty

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
