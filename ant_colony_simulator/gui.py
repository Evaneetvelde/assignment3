import pygame
import sys
import time
import argparse

from environment import Environment, TerrainType, Direction
from utils import create_environment, add_ants

# Colors - using the exact same colors as in improved_ant.py
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
ANT_COLOR = (68, 0, 8)
ANT_WITH_FOOD_COLOR = (158, 55, 17)

# Dirt and pheromone colors - exact match to improved_ant.py
DIRT_R, DIRT_G, DIRT_B = 217, 165, 78
DIRT_COLOR = (DIRT_R, DIRT_G, DIRT_B)
FOOD_COLOR = (158, 55, 17)
HOME_R, HOME_G, HOME_B = 96, 85, 33
FOOD_R, FOOD_G, FOOD_B = 255, 255, 255


class AntSimulationGUI:
    def __init__(
        self,
        environment: Environment,
        fps: int = 30,
        window_width: int = 800,
        window_height: int = 600,
        max_steps: int = 0,
        time_limit: float = 0,  # Time limit in seconds, 0 means no limit
        verbose: bool = True,
        progress_interval: int = 100,
    ):
        self.environment = environment
        self.cell_size = 1  # Internal surface is always 1px per cell
        self.fps = fps
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.paused = False
        self.show_pheromones = True
        self.step_count = 0
        self.show_stats = True
        self.show_grid = False
        self.simulation_complete = False
        self.start_time = None  # Will be set when simulation starts
        self.verbose = verbose
        self.progress_interval = progress_interval
        self.initial_food = environment.initial_food_amount

        pygame.init()
        self.window_width = window_width
        self.window_height = window_height

        # Stats panel on the right side — grid area is the full window height,
        # which is nearly square for typical square environments.
        self.stats_width = 220
        grid_area_w = window_width - self.stats_width
        grid_area_h = window_height
        scale = min(grid_area_w / environment.width, grid_area_h / environment.height)

        # Grid display dimensions and letterbox offsets to center the grid
        self.grid_display_w = round(scale * environment.width)
        self.grid_display_h = round(scale * environment.height)
        self.grid_offset_x = (grid_area_w - self.grid_display_w) // 2
        self.grid_offset_y = (grid_area_h - self.grid_display_h) // 2

        print(
            f"Environment size: {environment.width}x{environment.height} | "
            f"Cell scale: {scale:.2f}px | "
            f"Grid display: {self.grid_display_w}x{self.grid_display_h} | "
            f"Window: {window_width}x{window_height}"
        )
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Ant Colony Simulation")

        self.main_surface = pygame.Surface((environment.width, environment.height))
        self.main_surface.fill(DIRT_COLOR)

        self.font = pygame.font.SysFont("Arial", 15)
        self.font_small = pygame.font.SysFont("Arial", 13)
        self.clock = pygame.time.Clock()

        # Track last known positions for incremental updates
        self.last_food_positions = set()
        self.last_home_pheromones = set()
        self.last_food_pheromones = set()

    def run(self) -> None:
        running = True
        last_update = time.time()
        self.start_time = time.time()  # Record when simulation starts

        if self.verbose:
            print(f"Starting simulation with {len(self.environment.ants)} ants")
            print(f"Initial food amount: {self.initial_food}")
            print(f"Colony positions: {self.environment.colony_positions}")
            if self.time_limit > 0:
                print(f"Time limit: {self.time_limit} seconds")
            else:
                print("No time limit (unlimited)")

            if self.max_steps > 0:
                print(f"Max steps: {self.max_steps}")
            else:
                print("No step limit (unlimited)")

        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_p:
                        self.show_pheromones = not self.show_pheromones
                    elif event.key == pygame.K_s:
                        self.show_stats = not self.show_stats
                    elif event.key == pygame.K_g:
                        # Toggle grid
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_n and self.paused:
                        # Single step when paused
                        self.environment.update()
                        self.step_count += 1
                        # Check if simulation is complete after manual step
                        if self.environment.is_complete():
                            self.simulation_complete = True
                            self.paused = True

            # Update simulation if not paused and not complete
            current_time = time.time()
            if (
                not self.paused
                and not self.simulation_complete
                and current_time - last_update > 1.0 / self.fps
            ):
                self.environment.update()
                self.step_count += 1
                last_update = current_time

                # Print progress updates at specified intervals
                if self.verbose and self.step_count % self.progress_interval == 0:
                    food_collected = self.environment.food_collected
                    completion_pct = (
                        (food_collected / self.initial_food * 100)
                        if self.initial_food > 0
                        else 0
                    )
                    ants_with_food = sum(
                        1 for ant in self.environment.ants if ant.has_food
                    )

                    print(
                        f"Step {self.step_count}: "
                        f"Food collected: {food_collected}/{self.initial_food} ({completion_pct:.1f}%) | "
                        f"Ants with food: {ants_with_food}/{len(self.environment.ants)}"
                    )

                # Check if simulation is complete
                if self.environment.is_complete():
                    self.simulation_complete = True
                    self.paused = True
                    if self.verbose:
                        print(
                            f"Simulation complete! Reached 90% food collected in {self.step_count} steps."
                        )

                # Check if we've reached time limit
                elapsed_time = current_time - self.start_time
                if self.time_limit > 0 and elapsed_time >= self.time_limit:
                    self.simulation_complete = True
                    self.paused = True
                    if self.verbose:
                        print(f"Time limit reached: {self.time_limit} seconds")
                    if self.environment.is_complete():
                        if self.verbose:
                            print("Simulation complete! Reached 90% food collected.")
                    else:
                        completion = self.environment.get_completion_percentage()
                        if self.verbose:
                            print(
                                f"Simulation incomplete. Completion: {completion:.1f}%"
                            )

                # Check if we've reached max steps
                if self.max_steps > 0 and self.step_count >= self.max_steps:
                    if self.verbose:
                        print(f"Reached maximum steps: {self.max_steps}")
                    if self.environment.is_complete():
                        if self.verbose:
                            print("Simulation completed successfully (>=90% food).")
                    else:
                        if self.verbose:
                            print(
                                f"Simulation ended without collecting all food. Food collected: {self.environment.food_collected}/{self.environment.initial_food_amount}"
                            )
                    running = False

            # Draw everything
            self.draw()

            # Cap framerate
            self.clock.tick(60)

        pygame.quit()

    def draw(self) -> None:
        self.screen.fill(BLACK)
        self.main_surface.fill(DIRT_COLOR)

        if self.show_grid:
            for x in range(0, self.environment.width, 10):
                pygame.draw.line(
                    self.main_surface,
                    (DIRT_R - 20, DIRT_G - 20, DIRT_B - 20),
                    (x, 0),
                    (x, self.environment.height),
                    1,
                )
            for y in range(0, self.environment.height, 10):
                pygame.draw.line(
                    self.main_surface,
                    (DIRT_R - 20, DIRT_G - 20, DIRT_B - 20),
                    (0, y),
                    (self.environment.width, y),
                    1,
                )

        if self.show_pheromones:
            self.render_pixel_perfect()
        else:
            self.render_basic_terrain()

        self.render_ants()

        scaled_surface = pygame.transform.scale(
            self.main_surface, (self.grid_display_w, self.grid_display_h)
        )
        self.screen.blit(scaled_surface, (self.grid_offset_x, self.grid_offset_y))

        if self.show_stats:
            self.draw_stats()

        pygame.display.flip()

    def render_basic_terrain(self) -> None:
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                terrain = self.environment.get_terrain(x, y)
                if terrain == TerrainType.FOOD:
                    pygame.draw.rect(
                        self.main_surface,
                        FOOD_COLOR,
                        (
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )
                elif terrain == TerrainType.COLONY:
                    pygame.draw.rect(
                        self.main_surface,
                        (HOME_R, HOME_G, HOME_B),
                        (
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )
                elif terrain == TerrainType.WALL:
                    pygame.draw.rect(
                        self.main_surface,
                        GRAY,
                        (
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )

    def render_pixel_perfect(self) -> None:
        max_pheromone = 100.0

        for y in range(self.environment.height):
            for x in range(self.environment.width):
                terrain = self.environment.get_terrain(x, y)
                if terrain == TerrainType.FOOD:
                    pygame.draw.rect(
                        self.main_surface,
                        FOOD_COLOR,
                        (
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )
                    continue

                # Draw colony
                if terrain == TerrainType.COLONY:
                    pygame.draw.rect(
                        self.main_surface,
                        (HOME_R, HOME_G, HOME_B),
                        (
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )
                    continue

                # Draw walls
                if terrain == TerrainType.WALL:
                    pygame.draw.rect(
                        self.main_surface,
                        GRAY,
                        (
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )
                    continue

                # Get pheromone values
                home_val = self.environment.home_pheromones.get_value(x, y)
                food_val = self.environment.food_pheromones.get_value(x, y)

                # Skip pixels with no pheromones
                if home_val == 0 and food_val == 0:
                    continue

                # Calculate percentages
                home_pct = min(1.0, home_val / max_pheromone)
                food_pct = min(1.0, food_val / max_pheromone)

                # Calculate blended color, exactly like in improved_ant.py
                pixel_r = int(HOME_R * home_pct + DIRT_R * (1 - home_pct))
                pixel_g = int(HOME_G * home_pct + DIRT_G * (1 - home_pct))
                pixel_b = int(HOME_B * home_pct + DIRT_B * (1 - home_pct))

                pixel_r = int(FOOD_R * food_pct + pixel_r * (1 - food_pct))
                pixel_g = int(FOOD_G * food_pct + pixel_g * (1 - food_pct))
                pixel_b = int(FOOD_B * food_pct + pixel_b * (1 - food_pct))

                # Draw the pixel
                pygame.draw.rect(
                    self.main_surface,
                    (pixel_r, pixel_g, pixel_b),
                    (
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

    def render_ants(self) -> None:
        for ant in self.environment.ants:
            ant_colors = getattr(ant, "color", None)
            if ant_colors is not None:
                color = ant_colors[1] if ant.has_food else ant_colors[0]
            else:
                color = FOOD_COLOR if ant.has_food else ANT_COLOR

            this_x_f = ant.x * self.cell_size
            this_y_f = ant.y * self.cell_size

            dx, dy = Direction.get_delta(ant.direction)

            if abs(dx) > abs(dy):
                pygame.draw.rect(
                    self.main_surface,
                    color,
                    (this_x_f, this_y_f, 3 * self.cell_size, 2 * self.cell_size),
                )
            else:
                pygame.draw.rect(
                    self.main_surface,
                    color,
                    (this_x_f, this_y_f, 2 * self.cell_size, 3 * self.cell_size),
                )

    def draw_stats(self) -> None:
        panel_x = self.window_width - self.stats_width
        # Background and separator
        pygame.draw.rect(self.screen, (20, 20, 20), (panel_x, 0, self.stats_width, self.window_height))
        pygame.draw.line(self.screen, (80, 80, 80), (panel_x, 0), (panel_x, self.window_height), 1)

        total_ants = len(self.environment.ants)
        ants_with_food = sum(1 for ant in self.environment.ants if ant.has_food)
        food_collected = self.environment.food_collected
        total_food = self.environment.initial_food_amount
        completion_pct = (food_collected / total_food * 100) if total_food > 0 else 0

        elapsed_time = 0.0
        remaining_time = "N/A"
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            if self.time_limit > 0:
                remaining_time = f"{max(0, self.time_limit - elapsed_time):.1f}s"

        status = "COMPLETE" if self.simulation_complete else "PAUSED" if self.paused else "RUNNING"
        status_color = (80, 255, 80) if self.simulation_complete else (255, 200, 0) if self.paused else (255, 255, 255)

        tx = panel_x + 10
        y = 12
        line_h = 18

        def heading(label):
            nonlocal y
            if y > 12:
                y += 4
            surf = self.font.render(label, True, (160, 160, 160))
            self.screen.blit(surf, (tx, y))
            y += line_h + 2

        def row(label, value, color=WHITE):
            nonlocal y
            self.screen.blit(self.font_small.render(label, True, (140, 140, 140)), (tx, y))
            self.screen.blit(self.font_small.render(str(value), True, color), (tx + 100, y))
            y += line_h

        heading("SIMULATION")
        row("Status", status, status_color)
        row("Step", self.step_count)
        row("Time", f"{elapsed_time:.1f}s")
        if self.time_limit > 0:
            row("Remaining", remaining_time)

        heading("ANTS")
        row("Total", total_ants)
        row("With food", ants_with_food)

        heading("FOOD")
        row("Collected", f"{food_collected}/{total_food}")
        row("Progress", f"{completion_pct:.1f}%")

        heading("DISPLAY")
        row("FPS", f"{self.clock.get_fps():.0f}")
        row("Pheromones", "ON" if self.show_pheromones else "OFF")
        row("Grid", "ON" if self.show_grid else "OFF")

        y += 8
        pygame.draw.line(self.screen, (60, 60, 60), (tx, y), (panel_x + self.stats_width - 10, y), 1)
        y += 8

        heading("CONTROLS")
        for label, key in [("Pause", "SPACE"), ("Pheromones", "P"),
                            ("Grid", "G"), ("Stats", "S"), ("Step", "N")]:
            self.screen.blit(self.font_small.render(key, True, (200, 200, 100)), (tx, y))
            self.screen.blit(self.font_small.render(label, True, (140, 140, 140)), (tx + 50, y))
            y += line_h


def main():
    parser = argparse.ArgumentParser(description="Ant Colony Simulation")
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
        help="Ant strategy (non_cooperative, cooperative, smart or filename) (default: non_cooperative)",
    )
    parser.add_argument(
        "--strategy-file",
        type=str,
        help="Python file containing custom ant strategy",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=800,
        help="GUI window width in pixels (default: 800)",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=600,
        help="GUI window height in pixels (default: 600)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target simulation FPS (default: 30)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum simulation steps (0 = unlimited) (default: 0) - command line value takes precedence over environment file",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=0,
        help="Time limit in seconds (0 = no limit) (default: 0) - command line value takes precedence over environment file",
    )
    parser.add_argument(
        "--strategy2",
        type=str,
        default=None,
        help="Second ant strategy (name or file path) run in parallel with --strategy",
    )
    parser.add_argument(
        "--strategy-file2",
        type=str,
        help="Python file containing the second custom ant strategy",
    )
    parser.add_argument(
        "--ants2",
        type=int,
        default=1,
        help="Number of ants using --strategy2 (default: 1)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N steps (default: 100)",
    )
    args = parser.parse_args()

    try:
        environment = create_environment(args.env, args.width, args.height)

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

        # Color palette: (color_empty, color_with_food) per strategy group
        STRATEGY_COLORS = [
            ((68, 0, 8),    (158, 55, 17)),   # dark red / brown  (original)
            ((0, 40, 160),  (40, 140, 230)),  # dark blue / light blue
            ((0, 110, 10),  (60, 200, 70)),   # dark green / light green
            ((130, 0, 130), (210, 70, 210)),  # dark purple / light purple
        ]

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
            color=STRATEGY_COLORS[0],
        )

        if has_second:
            add_ants(
                environment,
                args.strategy2 or "non_cooperative",
                args.strategy_file2,
                args.ants2,
                verbose=not args.quiet,
                color=STRATEGY_COLORS[1],
            )
        gui = AntSimulationGUI(
            environment,
            fps=args.fps,
            window_width=args.window_width,
            window_height=args.window_height,
            max_steps=max_steps,
            time_limit=time_limit,
            verbose=not args.quiet,
            progress_interval=args.progress_interval,
        )
        gui.run()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
