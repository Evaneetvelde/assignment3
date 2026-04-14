import os.path
import importlib.util
import inspect
import random
from typing import Optional, Type

from environment import Environment, EnvironmentBuilder
from ant import Ant, AntStrategy

# Built-in strategies 
from strategies.non_cooperative import NonCooperativeStrategy
from strategies.cooperative import CooperativeStrategy
from strategies.smart import SmartStrategy

from common import Direction


def load_strategy_from_file(filepath: str, verbose: bool = True) -> Type[AntStrategy]:
    if not os.path.exists(filepath):
        raise ValueError(f"Strategy file not found: {filepath}")

    module_name = os.path.splitext(os.path.basename(filepath))[0]

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {filepath}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    strategy_classes = []
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, AntStrategy) and obj != AntStrategy:
            strategy_classes.append(obj)

    if not strategy_classes:
        raise ValueError(f"No AntStrategy implementation found in {filepath}")

    if len(strategy_classes) > 1 and verbose:
        print(
            f"Warning: Multiple AntStrategy implementations found in {filepath}. Using {strategy_classes[0].__name__}"
        )

    return strategy_classes[0]


def create_environment(
    env_type: str, width: int, height: int, verbose: bool = True
) -> Environment:
    if env_type == "simple":
        return EnvironmentBuilder.create_simple(width, height)
    elif env_type == "obstacle":
        return EnvironmentBuilder.create_obstacle_course(width, height)
    elif env_type == "maze":
        return EnvironmentBuilder.create_maze(width, height)
    elif env_type == "empty":
        return EnvironmentBuilder.create_empty(width, height)
    elif os.path.isfile(env_type):
        try:
            return EnvironmentBuilder.load_from_file(env_type, verbose=verbose)
        except Exception as e:
            raise ValueError(
                f"Failed to load environment from file {env_type}: {str(e)}"
            )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def add_ants(
    environment: Environment,
    strategy_name: str,
    strategy_file: Optional[str],
    count: int,
    verbose: bool = True,
    color: Optional[tuple] = None,  # ((r,g,b) empty, (r,g,b) with_food)
) -> None:

    # Create strategy based on name or load from file
    strategy = None

    if strategy_file:
        try:
            # Load strategy class from file
            strategy_class = load_strategy_from_file(strategy_file, verbose=verbose)
            strategy = strategy_class()
            if verbose:
                print(
                    f"Loaded strategy '{strategy_class.__name__}' from {strategy_file}"
                )
        except Exception as e:
            raise ValueError(f"Error loading strategy from {strategy_file}: {str(e)}")

    # If strategy was loaded from file, check for set_environment hook
    if strategy_file and strategy is not None and hasattr(strategy, "set_environment"):
        strategy.set_environment(environment)

    # If no strategy loaded from file, use built-in strategy
    if strategy is None:
        if strategy_name == "non_cooperative":
            strategy = NonCooperativeStrategy()
        elif strategy_name == "cooperative":
            strategy = CooperativeStrategy()
        elif strategy_name == "smart":
            strategy = SmartStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    # Add ants at colony positions
    if not environment.colony_positions:
        raise ValueError("No colony positions in environment")

    # Create ants equally distributed across all colonies
    for i in range(count):
        colony_pos = environment.colony_positions[i % len(environment.colony_positions)]
        x, y = colony_pos
        # Create ant with globally unique id across all add_ants calls
        direction = random.choice(list(Direction))
        ant = Ant(x, y, direction, strategy, ant_id=environment.next_ant_id)
        environment.next_ant_id += 1
        if color is not None:
            ant.color = color  # ((r,g,b) empty, (r,g,b) with_food)
        environment.add_ant(ant)
