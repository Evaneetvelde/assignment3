from typing import Optional
from ant import Ant
import random
import math
from common import (
    TerrainType,
    Direction,
    AntPerception,
    AntAction,
)


# Class for pheromone handling
class PheromoneMap:
    def __init__(self, width: int, height: int, evaporation_rate: float = 0.999):
        self.width = width
        self.height = height
        self.evaporation_rate = evaporation_rate
        # Use a dictionary for sparse representation of pheromones
        # Key is (x, y) tuple, value is pheromone strength
        self.values = {}
        self.modified_positions = set()

    def add_pheromone(self, x: int, y: int, amount: float) -> None:
        """Add pheromone at position (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            pos = (x, y)
            # Add maximum pheromone amount between current and new amount
            self.values[pos] = max(self.values.get(pos, 0), amount)
            self.modified_positions.add(pos)

    def get_value(self, x: int, y: int) -> float:
        """Get pheromone value at position (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.values.get((x, y), 0.0)
        return 0.0

    def evaporate(self) -> None:
        """Evaporate pheromones"""
        # Create a list of positions to potentially remove
        positions_to_remove = []

        # Update all pheromone values
        for pos, value in self.values.items():
            # Apply evaporation
            new_value = value * self.evaporation_rate

            # If value is very small, mark for removal
            if new_value < 0.01:
                positions_to_remove.append(pos)
            else:
                self.values[pos] = new_value
                self.modified_positions.add(pos)

        # Remove very small values
        for pos in positions_to_remove:
            del self.values[pos]

    def get_strongest_direction(
        self, x: int, y: int, vision_range: int = 3
    ) -> Optional[Direction]:
        """Get direction with highest pheromone concentration"""
        max_value = 0.0
        best_direction = None

        for direction in Direction:
            dx, dy = Direction.get_delta(direction)
            value_sum = 0.0

            for strength in range(1, vision_range + 1):
                check_x, check_y = x + dx * strength, y + dy * strength
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    value_sum += (
                        self.get_value(int(check_x), int(check_y)) / strength
                    )  # Closer is stronger

            if value_sum > max_value:
                max_value = value_sum
                best_direction = direction

        return best_direction


# Environment class to represent the world
class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [
            [TerrainType.EMPTY.value for _ in range(width)] for _ in range(height)
        ]
        self.food_amounts = [[0 for _ in range(width)] for _ in range(height)]
        self.home_pheromones = PheromoneMap(width, height)
        self.food_pheromones = PheromoneMap(width, height)
        self.ants = []
        self.colony_positions = []
        self.colony_radius = 2  # This creates a 5x5 area (radius 2 around center point)
        self.food_positions = set()
        self.initial_food_amount = 0
        self.food_removed = 0
        self.food_collected = 0
        self.steps = 0
        self.pheromones_enabled = True
        self.next_ant_id = 1  # For tracking sequential ant IDs

    def disable_pheromones(self) -> None:
        self.pheromones_enabled = False
        self.home_pheromones = PheromoneMap(self.width, self.height)
        self.food_pheromones = PheromoneMap(self.width, self.height)

    def add_wall(self, x: int, y: int) -> None:
        if self.is_valid_position(x, y):
            self.grid[y][x] = TerrainType.WALL.value

    def add_food(self, x: int, y: int, amount: int = 1) -> None:
        if self.is_valid_position(x, y) and self.grid[y][x] == TerrainType.EMPTY.value:
            self.grid[y][x] = TerrainType.FOOD.value
            self.food_amounts[y][x] += amount
            self.food_positions.add((x, y))
            self.initial_food_amount += amount

    def add_food_area(
        self, x: int, y: int, width: int, height: int, amount: int = 1
    ) -> None:
        for i in range(width):
            for j in range(height):
                self.add_food(x + i, y + j, amount)

    def remove_food(self, x: int, y: int) -> bool:
        if (
            self.is_valid_position(x, y)
            and self.grid[y][x] == TerrainType.FOOD.value
            and self.food_amounts[y][x] > 0
        ):
            self.food_amounts[y][x] -= 1

            if self.food_amounts[y][x] == 0:
                self.grid[y][x] = TerrainType.EMPTY.value
                self.food_positions.discard((x, y))

            self.food_removed += 1
            return True

        return False

    def add_colony(self, x: int, y: int) -> None:
        if self.is_valid_position(x, y) and self.grid[y][x] == TerrainType.EMPTY.value:
            self.grid[y][x] = TerrainType.COLONY.value
            self.colony_positions.append((x, y))

    def add_ant(self, ant) -> None:
        self.ants.append(ant)

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return (
            self.is_valid_position(x, y) and self.grid[y][x] != TerrainType.WALL.value
        )

    def get_terrain(self, x: int, y: int) -> Optional[TerrainType]:
        if self.is_valid_position(x, y):
            # Check if the position is within the colony radius
            for colony_x, colony_y in self.colony_positions:
                # Check if the position is within the colony radius
                if (
                    abs(x - colony_x) <= self.colony_radius
                    and abs(y - colony_y) <= self.colony_radius
                ):
                    if self.grid[y][x] == TerrainType.FOOD.value:
                        return TerrainType.FOOD
                    elif self.grid[y][x] == TerrainType.WALL.value:
                        return TerrainType.WALL
                    return TerrainType.COLONY

            # If not in colony radius, return the actual terrain
            return TerrainType(self.grid[y][x])
        return None

    def update(self) -> None:
        if self.pheromones_enabled:
            self.home_pheromones.evaporate()
            self.food_pheromones.evaporate()
        for ant in self.ants:
            perception = self.get_perception_for_ant(ant)
            action = ant.decide_action(perception)
            self.execute_action(ant, action)

        self.steps += 1

    def get_perception_for_ant(self, ant: Ant) -> AntPerception:

        perception = AntPerception()
        perception.has_food = ant.has_food
        perception.direction = ant.direction
        perception.home_pheromone_level = ant.home_pheromone
        perception.food_pheromone_level = ant.food_pheromone
        perception.pheromone_decrease_rate = ant.pheromone_decrease_rate
        perception.food_collected = ant.food_collected
        perception.steps_taken = ant.steps_taken
        perception.ant_id = ant.id

        current_terrain = self.get_terrain(int(ant.x), int(ant.y))
        if current_terrain is not None:
            perception.visible_cells[(0, 0)] = current_terrain

        for dx in range(-ant.vision_range, ant.vision_range + 1):
            for dy in range(-ant.vision_range, ant.vision_range + 1):
                if dx == 0 and dy == 0:
                    continue

                check_x = int(ant.x + dx)
                check_y = int(ant.y + dy)

                # Check if the point is within the ant's vision field based on distance and angle

                # Calculate distance
                distance = math.sqrt(dx * dx + dy * dy)

                # If point is too far, it's not in vision field
                if distance > ant.vision_range:
                    continue

                # Get ant's direction vector
                dir_dx, dir_dy = Direction.get_delta(ant.direction)

                # Calculate angle between ant's direction and point
                # First normalize vectors
                if distance > 0:
                    point_dx, point_dy = dx / distance, dy / distance
                else:
                    # If distance is 0, point is at ant's position (already handled with continue above)
                    continue

                dir_magnitude = math.sqrt(dir_dx * dir_dx + dir_dy * dir_dy)
                if dir_magnitude > 0:
                    dir_dx, dir_dy = dir_dx / dir_magnitude, dir_dy / dir_magnitude

                # Calculate dot product
                dot_product = dir_dx * point_dx + dir_dy * point_dy

                # Clamp to valid range for acos
                dot_product = max(-1.0, min(1.0, dot_product))

                # Calculate angle in degrees
                angle = math.degrees(math.acos(dot_product))

                # Check if angle is within vision field
                half_vision_angle = ant.vision_angle / 2
                if angle > half_vision_angle:
                    continue

                is_blocked = False

                # Simple line-of-sight check
                if (
                    abs(dx) > 1 or abs(dy) > 1
                ):  # Skip line-of-sight check for adjacent cells (they're always visible)
                    # Calculate steps for line-of-sight check
                    steps = max(abs(dx), abs(dy))
                    step_x = dx / steps
                    step_y = dy / steps

                    # Check each step along the line of sight
                    for step in range(1, steps):
                        check_step_x = int(ant.x + step * step_x)
                        check_step_y = int(ant.y + step * step_y)

                        if (
                            self.is_valid_position(check_step_x, check_step_y)
                            and self.grid[check_step_y][check_step_x]
                            == TerrainType.WALL.value
                        ):
                            is_blocked = True
                            break

                if is_blocked:
                    continue

                # If valid position and not blocked, add to visible cells
                if self.is_valid_position(check_x, check_y):
                    terrain = self.grid[check_y][check_x]
                    # Convert integer value to TerrainType enum for consistency
                    perception.visible_cells[(dx, dy)] = TerrainType(terrain)

                    # Also add pheromone information
                    perception.food_pheromone[(dx, dy)] = (
                        self.food_pheromones.get_value(check_x, check_y)
                    )
                    perception.home_pheromone[(dx, dy)] = (
                        self.home_pheromones.get_value(check_x, check_y)
                    )

                    # Check for other ants
                    for other_ant in self.ants:
                        if (
                            other_ant != ant
                            and int(other_ant.x) == check_x
                            and int(other_ant.y) == check_y
                        ):
                            perception.nearby_ants.append(
                                ((dx, dy), other_ant.has_food)
                            )
                            break
        return perception

    def execute_action(self, ant: "Ant", action: "AntAction") -> bool:
        if action == AntAction.MOVE_FORWARD:
            dx, dy = Direction.get_delta(ant.direction)
            new_x, new_y = ant.x + dx, ant.y + dy

            success = self.is_walkable(int(new_x), int(new_y))
            ant.move_forward(success)
            return success

        elif action == AntAction.TURN_LEFT:
            ant.turn_left()
            return True

        elif action == AntAction.TURN_RIGHT:
            ant.turn_right()
            return True

        elif action == AntAction.PICK_UP_FOOD:
            if (
                not ant.has_food
                and self.get_terrain(int(ant.x), int(ant.y)) == TerrainType.FOOD
            ):
                success = self.remove_food(int(ant.x), int(ant.y))
                ant.pick_up_food(success)

                # NOTE: Automatic pheromone deposition could be implemented here like this:
                # if success:
                #    amount = ant.deposit_pheromone()
                #    self.food_pheromones.add_pheromone(int(ant.x), int(ant.y), amount)
                # This would automatically deposit food pheromones whenever an ant picks up food,
                # without requiring the strategy to explicitly choose the DEPOSIT_FOOD_PHEROMONE action.

                return success
            else:
                ant.pick_up_food(False)
                return False

        elif action == AntAction.DROP_FOOD:
            if (
                ant.has_food
                and self.get_terrain(int(ant.x), int(ant.y)) == TerrainType.COLONY
            ):
                self.food_collected += 1
                ant.drop_food(True)

                # NOTE: Similar automatic pheromone deposition could be implemented here:
                # amount = ant.deposit_pheromone()
                # self.home_pheromones.add_pheromone(int(ant.x), int(ant.y), amount)

                return True
            else:
                ant.drop_food(False)
                return False

        elif action == AntAction.DEPOSIT_HOME_PHEROMONE:
            if self.pheromones_enabled:
                amount = ant.deposit_pheromone()
                self.home_pheromones.add_pheromone(int(ant.x), int(ant.y), amount)
                return True
            return False

        elif action == AntAction.DEPOSIT_FOOD_PHEROMONE:
            if self.pheromones_enabled:
                amount = ant.deposit_pheromone()
                self.food_pheromones.add_pheromone(int(ant.x), int(ant.y), amount)
                return True
            return False

        elif action == AntAction.NO_ACTION:
            return True

        return False

    def is_complete(self) -> bool:
        if self.initial_food_amount <= 0:
            return False
        collected_pct = self.food_removed / self.initial_food_amount
        returned_pct = self.food_collected / self.initial_food_amount
        return collected_pct >= 0.75 and returned_pct >= 0.50

    def get_completion_percentage(self) -> float:
        if self.initial_food_amount == 0:
            return 0.0
        return self.food_collected / self.initial_food_amount * 100.0


# Environment Builder to create different scenarios
class EnvironmentBuilder:
    @staticmethod
    def create_empty(width: int, height: int) -> Environment:
        return Environment(width, height)

    @staticmethod
    def create_simple(width: int, height: int) -> Environment:
        env = Environment(width, height)

        center_x, center_y = width // 2, height // 2
        env.add_colony(center_x, center_y)

        margin = min(width, height) // 10

        env.add_food_area(margin, margin, 5, 5)
        env.add_food_area(width - margin - 5, margin, 5, 5)
        env.add_food_area(margin, height - margin - 5, 5, 5)
        env.add_food_area(width - margin - 5, height - margin - 5, 5, 5)

        return env

    @staticmethod
    def create_obstacle_course(width: int, height: int) -> Environment:
        env = Environment(width, height)

        center_x, center_y = width // 2, height // 2
        env.add_colony(center_x, center_y)

        food_margin = 10
        env.add_food_area(food_margin, height // 4, 8, 8)
        env.add_food_area(width - food_margin - 8, height // 4, 8, 8)
        env.add_food_area(width - food_margin - 8, height - height // 4 - 8, 8, 8)
        env.add_food_area(food_margin, height - height // 4 - 8, 8, 8)

        wall_x = width // 2
        gap_y = height // 2
        gap_size = height // 10

        for y in range(height):
            if abs(y - gap_y) > gap_size:
                env.add_wall(wall_x, y)

        wall_y = height // 2
        gap_x = width // 2
        gap_size = width // 10

        for x in range(width):
            if abs(x - gap_x) > gap_size:
                env.add_wall(x, wall_y)

        return env

    @staticmethod
    def create_maze(width: int, height: int) -> Environment:
        env = Environment(width, height)

        center_x, center_y = width // 2, height // 2
        env.add_colony(center_x, center_y)

        cell_size = 20
        for x in range(0, width, cell_size):
            for y in range(0, height, cell_size):
                if random.random() < 0.3 and (
                    abs(x - center_x) > cell_size or abs(y - center_y) > cell_size
                ):
                    wall_len = random.randint(5, cell_size - 2)
                    if random.random() < 0.5:
                        for i in range(wall_len):
                            if x + i < width:
                                env.add_wall(x + i, y)
                    else:
                        for i in range(wall_len):
                            if y + i < height:
                                env.add_wall(x, y + i)

        for _ in range(5):
            while True:
                fx = random.randint(0, width - 6)
                fy = random.randint(0, height - 6)

                if math.sqrt((fx - center_x) ** 2 + (fy - center_y) ** 2) > width // 4:
                    env.add_food_area(fx, fy, 5, 5)
                    break

        return env

    @staticmethod
    def load_from_file(filename: str, verbose: bool = True) -> Optional[Environment]:
        """Load environment configuration from file

        The file format supports the following sections:
        - DIMENSIONS: width height
        - WALL: x y (for adding wall/obstacle positions)
        - FOOD: x y [amount] (for adding food with optional specific amounts)
        - COLONY: x y (for adding colony positions)
        - ANTS: count (for specifying the number of ants to create)
        - TIME_LIMIT: seconds (for specifying simulation time limit in seconds)
        - MAX_STEPS: steps (for specifying maximum simulation steps)

        Example:
        ```
        DIMENSIONS: 100 100
        WALL:
        10 10
        10 11
        10 12
        FOOD:
        50 30 5
        70 80 10
        COLONY:
        10 20
        ANTS:
        50
        TIME_LIMIT:
        60
        MAX_STEPS:
        10000
        ```
        """
        try:
            with open(filename, "r") as f:
                lines = f.readlines()

                width, height = 100, 100
                env = None
                current_section = None
                ant_count = 0
                time_limit = 0  # Default: no time limit
                max_steps = 0  # Default: no step limit

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if line.endswith(":"):
                        current_section = line[:-1].upper()
                        continue

                    if current_section == "DIMENSIONS":
                        width, height = map(int, line.split())
                        if verbose:
                            print(
                                f"Loading environment with dimensions: {width}x{height}"
                            )
                        env = Environment(width, height)
                    elif current_section == "WALL" and env is not None:
                        parts = line.split()
                        if len(parts) >= 2:
                            x, y = int(parts[0]), int(parts[1])
                            if env.is_valid_position(x, y):
                                env.add_wall(x, y)
                    elif current_section == "FOOD" and env is not None:
                        parts = line.split()
                        if len(parts) >= 3:
                            x, y = int(parts[0]), int(parts[1])
                            amount = int(parts[2])
                            if env.is_valid_position(x, y):
                                env.add_food(x, y, amount)
                        elif len(parts) >= 2:
                            x, y = int(parts[0]), int(parts[1])
                            if env.is_valid_position(x, y):
                                env.add_food(x, y)
                    elif current_section == "COLONY" and env is not None:
                        parts = line.split()
                        if len(parts) >= 2:
                            x, y = int(parts[0]), int(parts[1])
                            if env.is_valid_position(x, y):
                                env.add_colony(x, y)
                    elif current_section == "ANTS" and env is not None:
                        try:
                            ant_count = int(line.strip())
                            if verbose:
                                print(f"Loading environment with {ant_count} ants")
                        except ValueError:
                            if verbose:
                                print(f"Invalid ant count: {line}")
                    elif current_section == "TIME_LIMIT":
                        try:
                            time_limit = int(line.strip())
                            if verbose:
                                print(f"Environment time limit: {time_limit} seconds")
                        except ValueError:
                            if verbose:
                                print(f"Invalid time limit: {line}")
                    elif current_section == "MAX_STEPS":
                        try:
                            max_steps = int(line.strip())
                            if verbose:
                                print(f"Environment max steps: {max_steps}")
                        except ValueError:
                            if verbose:
                                print(f"Invalid max steps: {line}")

                if env is None:
                    env = Environment(width, height)

                if not env.colony_positions:
                    if verbose:
                        print(
                            "Warning: No colony positions defined in environment file. Adding one at center."
                        )
                    env.add_colony(env.width // 2, env.height // 2)

                # Store the ant count as an attribute of the environment for later use
                env.requested_ant_count = ant_count

                # Store time limit and max steps as environment attributes
                env.time_limit = time_limit
                env.max_steps = max_steps

                return env
        except Exception as e:
            if verbose:
                print(f"Error loading environment from file: {e}")
            return None

    @staticmethod
    def save_to_file(env: Environment, filename: str) -> bool:
        try:
            with open(filename, "w") as f:
                f.write(f"DIMENSIONS:\n{env.width} {env.height}\n\n")

                if env.colony_positions:
                    f.write("COLONY:\n")
                    for x, y in env.colony_positions:
                        f.write(f"{x} {y}\n")
                    f.write("\n")

                if env.food_positions:
                    f.write("FOOD:\n")
                    for x, y in env.food_positions:
                        amount = env.food_amounts[y][x]
                        f.write(f"{x} {y} {amount}\n")
                    f.write("\n")

                walls = []
                for y in range(env.height):
                    for x in range(env.width):
                        if env.grid[y][x] == TerrainType.WALL.value:
                            walls.append((x, y))

                if walls:
                    f.write("WALL:\n")
                    for x, y in walls:
                        f.write(f"{x} {y}\n")
                    f.write("\n")

                # Write the number of ants
                f.write("ANTS:\n")
                f.write(f"{len(env.ants) if env.ants else 0}\n\n")

                # Write time limit if it's set
                if hasattr(env, "time_limit") and env.time_limit > 0:
                    f.write("TIME_LIMIT:\n")
                    f.write(f"{env.time_limit}\n\n")

                # Write max steps if it's set
                if hasattr(env, "max_steps") and env.max_steps > 0:
                    f.write("MAX_STEPS:\n")
                    f.write(f"{env.max_steps}\n\n")

                return True
        except Exception as e:
            print(f"Error saving environment to file: {e}")
            return False
