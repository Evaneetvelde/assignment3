# Command-line simulation runner for ant colony simulation.

import argparse
import time
import sys

from environment import Environment
from utils import create_environment, add_ants


class SimulationRunner:
    def __init__(
        self,
        environment: Environment,
        max_steps: int = 10000,
        progress_interval: int = 100,
        time_limit: float = 0,  # Time limit in seconds, 0 means no limit
    ):
        self.environment = environment
        self.max_steps = max_steps
        self.progress_interval = progress_interval
        self.step_count = 0
        self.time_limit = time_limit
        self.duration = 0.0  # Initialize duration attribute

    def run(self, verbose: bool = True) -> dict:
        start_time = time.time()
        initial_food = self.environment.initial_food_amount
        strategy_names = sorted(
            {
                ant.strategy.get_name()
                if ant.strategy and hasattr(ant.strategy, "get_name")
                else ant.strategy.__class__.__name__
                for ant in self.environment.ants
            }
        )
        show_strategy_split = len(strategy_names) == 2
        if verbose:
            print(f"Starting simulation with {len(self.environment.ants)} ants")
            print(f"Initial food amount: {initial_food}")
            print(f"Colony positions: {self.environment.colony_positions}")
            print(f"Environment time limit: {self.time_limit} seconds")
            print(f"Environment max steps: {self.max_steps} steps")

            if self.time_limit > 0:
                print(f"Time limit: {self.time_limit} seconds")
            else:
                print("No time limit (unlimited)")

            if self.max_steps > 0:
                print(f"Max steps: {self.max_steps}")
            else:
                print("No step limit (unlimited)")

        # Run until all food is collected or max steps reached or time limit exceeded
        while (
            not self.environment.is_complete()
            and (self.max_steps <= 0 or self.step_count < self.max_steps)
            and (self.time_limit <= 0 or time.time() - start_time < self.time_limit)
        ):
            self.environment.update()
            self.step_count += 1
            # print(f"Step {self.step_count} / {self.max_steps}")
            # Print progress updates at specified intervals
            if verbose and self.step_count % self.progress_interval == 0:
                food_collected = self.environment.food_collected
                completion_pct = (
                    (food_collected / initial_food * 100) if initial_food > 0 else 0
                )
                ants_with_food = sum(1 for ant in self.environment.ants if ant.has_food)

                if show_strategy_split:
                    counts = {name: 0 for name in strategy_names}
                    for ant in self.environment.ants:
                        if ant.has_food:
                            name = (
                                ant.strategy.get_name()
                                if ant.strategy and hasattr(ant.strategy, "get_name")
                                else ant.strategy.__class__.__name__
                            )
                            counts[name] += 1
                    split = f"{strategy_names[0]}: {counts[strategy_names[0]]}, {strategy_names[1]}: {counts[strategy_names[1]]}"
                    ants_with_food_text = (
                        f"{ants_with_food}/{len(self.environment.ants)} ({split})"
                    )
                else:
                    ants_with_food_text = (
                        f"{ants_with_food}/{len(self.environment.ants)}"
                    )

                print(
                    f"Step {self.step_count}: "
                    f"Food collected: {food_collected}/{initial_food} ({completion_pct:.1f}%) | "
                    f"Ants with food: {ants_with_food_text}"
                )

        # Print final results
        end_time = time.time()
        self.duration = end_time - start_time

        if verbose:
            if self.environment.is_complete():
                print(
                    f"\nSimulation complete! Reached 90% food collected in {self.step_count} steps."
                )
            elif self.time_limit > 0 and time.time() - start_time >= self.time_limit:
                print(f"\nTime limit reached ({self.time_limit} seconds).")
                print(
                    f"Food collected: {self.environment.food_collected}/{initial_food} "
                    f"({self.environment.food_collected / initial_food * 100:.1f}%)"
                )
            else:
                print(
                    f"\nSimulation reached max steps ({self.max_steps}) without collecting all food."
                )
                print(
                    f"Food collected: {self.environment.food_collected}/{initial_food} "
                    f"({self.environment.food_collected / initial_food * 100:.1f}%)"
                )

            print(f"Total runtime: {self.duration:.2f} seconds")
            print(f"Average steps per second: {self.step_count / self.duration:.1f}")

        # Calculate completion percentage
        completion_percentage = (
            (self.environment.food_collected / initial_food * 100)
            if initial_food > 0
            else 0
        )

        # Return a dictionary with completion percentage, number of steps, and time taken
        success = self.environment.is_complete()
        return {
            "completion_percentage": completion_percentage,
            "food_collected": self.environment.food_collected,
            "food_removed": getattr(self.environment, "food_removed", 0),
            "total_food": self.environment.initial_food_amount,
            "time_limit": self.time_limit,
            "max_steps": self.max_steps,
            "steps": self.step_count,
            "time_taken": self.duration,
            "success": success,
        }


def main():
    parser = argparse.ArgumentParser(description="Run ant colony simulation (headless)")

    parser.add_argument(
        "--env",
        type=str,
        default="simple",
        help="Environment type (simple, obstacle, maze) or path to environment file (e.g., simple_env.txt)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Environment width (default: 100) - ignored when loading from file",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=100,
        help="Environment height (default: 100) - ignored when loading from file",
    )

    parser.add_argument(
        "--ants",
        type=int,
        default=10,
        help="Number of ants (default: 10) - overridden by ANTS section in environment file if present",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="non_cooperative",
        help="Ant strategy to use (non_cooperative, cooperative, smart) (default: non_cooperative)",
    )
    parser.add_argument(
        "--strategy-file",
        type=str,
        help="Path to Python file containing custom ant strategy",
    )

    parser.add_argument(
        "--strategy2",
        type=str,
        default=None,
        help="Second ant strategy (runs alongside the first group)",
    )
    parser.add_argument(
        "--ants2",
        type=int,
        default=10,
        help="Number of ants for the second strategy (default: 10)",
    )
    parser.add_argument(
        "--strategy-file2",
        type=str,
        default=None,
        help="Path to Python file containing the second custom ant strategy",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum simulation steps (default: 0, no limit) - command line value takes precedence over environment file",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="How often to print progress updates (in steps) (default: 100)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=0,
        help="Time limit for simulation (in seconds) (default: 0, no limit) - command line value takes precedence over environment file",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    try:
        environment = create_environment(
            args.env, args.width, args.height, verbose=not args.quiet
        )

        # Check if environment file specified a number of ants
        ant_count = args.ants
        if (
            hasattr(environment, "requested_ant_count")
            and environment.requested_ant_count > 0
        ):
            ant_count = environment.requested_ant_count
            if not args.quiet:
                print(f"Using ant count from environment file: {ant_count}")

        # For time limit and max steps, command line args take precedence
        time_limit = args.time_limit

        if (
            time_limit == 0
            and hasattr(environment, "time_limit")
            and environment.time_limit > 0
        ):
            # Only use environment time_limit if command line didn't specify a value
            time_limit = environment.time_limit
            if not args.quiet:
                print(f"Using time limit from environment file: {time_limit} seconds")

        max_steps = args.max_steps
        if (
            max_steps == 0
            and hasattr(environment, "max_steps")
            and environment.max_steps > 0
        ):
            # Only use environment max_steps if command line didn't specify a value
            max_steps = environment.max_steps
            if not args.quiet:
                print(f"Using max steps from environment file: {max_steps} steps")

        has_second = args.strategy2 is not None or args.strategy_file2 is not None
        first_count = ant_count - args.ants2 if has_second else ant_count

        if first_count < 0:
            raise ValueError(
                f"--ants2 ({args.ants2}) exceeds total ant count ({ant_count})"
            )

        add_ants(
            environment,
            args.strategy,
            args.strategy_file,
            first_count,
            verbose=not args.quiet,
        )

        if has_second:
            add_ants(
                environment,
                args.strategy2 or "non_cooperative",
                args.strategy_file2,
                args.ants2,
                verbose=not args.quiet,
            )
        runner = SimulationRunner(
            environment,
            max_steps=max_steps,
            progress_interval=args.progress_interval,
            time_limit=time_limit,
        )

        result = runner.run(verbose=not args.quiet)

        if not args.quiet:
            print(f"\nSimulation completed in {result['steps']} steps")
            print(
                f"Food collected: {environment.food_collected}/{environment.initial_food_amount} ({result['completion_percentage']:.1f}%)"
            )

        # Return the result dictionary instead of an integer
        return result

    except Exception as e:
        error_message = f"Error: {str(e)}"

        # Try to get time_limit and max_steps values if they were set earlier, otherwise use 0
        safe_time_limit = time_limit if "time_limit" in locals() else 0
        safe_max_steps = max_steps if "max_steps" in locals() else 0

        error_result = {
            "error": str(e),
            "completion_percentage": 0,
            "steps": 0,
            "time_taken": 0,
            "food_collected": 0,
            "food_removed": 0,
            "total_food": 0,
            "time_limit": safe_time_limit,
            "max_steps": safe_max_steps,
            "success": False,
        }

        if not args.quiet:
            # Standard error output only if not quiet
            print(error_message, file=sys.stderr)

        return error_result


if __name__ == "__main__":
    result = main()
    # Return 0 for success (>=90% completion) or 1 otherwise
    exit_code = (
        0 if "error" not in result and result.get("success") else 1
    )
    sys.exit(exit_code)
