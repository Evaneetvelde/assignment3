# Ant Simulation Command-Line Arguments


## Simulation Mode (Headless)

```bash
usage: simulation.py [-h] [--env ENV] [--width WIDTH] [--height HEIGHT] [--ants ANTS] [--strategy STRATEGY] [--strategy-file STRATEGY_FILE]
                     [--max-steps MAX_STEPS] [--progress-interval PROGRESS_INTERVAL] [--time-limit TIME_LIMIT] [--quiet]
                     [--no-pheromones]

Run ant colony simulation (headless)

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Environment type (simple, obstacle, maze) or path to environment file (e.g., simple_env.txt)
  --width WIDTH         Environment width (default: 100) - ignored when loading from file
  --height HEIGHT       Environment height (default: 100) - ignored when loading from file
  --ants ANTS           Number of ants (default: 10) - overridden by ANTS section in environment file if present
  --strategy STRATEGY   Ant strategy to use (non_cooperative or custom) (default: non_cooperative)
  --strategy-file STRATEGY_FILE
                        Path to Python file containing custom ant strategy
  --max-steps MAX_STEPS
                        Maximum simulation steps (default: 0, no limit) - command line value takes precedence over environment file
  --progress-interval PROGRESS_INTERVAL
                        How often to print progress updates (in steps) (default: 100)
  --time-limit TIME_LIMIT
                        Time limit for simulation (in seconds) (default: 0, no limit) - command line value takes precedence over environment file
  --quiet               Suppress progress output
```

## GUI Mode

```bash
usage: gui.py [-h] [--env ENV] [--width WIDTH] [--height HEIGHT] [--ants ANTS] [--strategy STRATEGY] [--strategy-file STRATEGY_FILE]
              [--cell-size CELL_SIZE] [--scale SCALE] [--fps FPS] [--max-steps MAX_STEPS] [--time-limit TIME_LIMIT] [--quiet]
              [--progress-interval PROGRESS_INTERVAL] [--no-pheromones]

Ant Colony Simulation

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Environment type (simple, obstacle, maze) or path to environment file (e.g., simple_env.txt)
  --width WIDTH         Environment width (default: 100) - ignored when loading from file
  --height HEIGHT       Environment height (default: 100) - ignored when loading from file
  --ants ANTS           Number of ants (default: 10) - overridden by ANTS section in environment file if present
  --strategy STRATEGY   Ant strategy (non_cooperative or filename) (default: non_cooperative)
  --strategy-file STRATEGY_FILE
                        Python file containing custom ant strategy
  --cell-size CELL_SIZE
                        Pixel size for each cell (default: 1px = exact match to improved_ant.py)
  --scale SCALE         Display scale factor (default: 2x)
  --fps FPS             Target simulation FPS (default: 30)
  --max-steps MAX_STEPS
                        Maximum simulation steps (0 = unlimited) (default: 0) - command line value takes precedence over environment file
  --time-limit TIME_LIMIT
                        Time limit in seconds (0 = no limit) (default: 0) - command line value takes precedence over environment file
  --quiet               Suppress progress output
  --progress-interval PROGRESS_INTERVAL
                        Print progress every N steps (default: 100)
```

## Key Differences

1. **GUI-Specific Arguments**:
   - `--cell-size`: Controls the pixel size for each cell in the visualization
   - `--scale`: Controls the display scale factor (For small environments increase scale to show more details)
   - `--fps`: Sets the target simulation frame rate (Won't perfectly match but will try)

2. **Default Values**:
   - `--max-steps`: Both modes default to 0 (unlimited)
   - `--time-limit`: Both modes default to 0 (unlimited)

## Note on Environment Files

When using environment files (via the `--env` argument with a file path), the following behavior applies:

1. **Width and Height**: The dimensions specified in the environment file take precedence over the command-line arguments.

2. **Ant Count**: If the environment file contains an `ANTS` section with a count value, this will override the `--ants` command-line argument.

3. **Time Limit and Max Steps**: If the environment file contains `TIME_LIMIT` or `MAX_STEPS` sections, these will be used as default values but can be overridden by the corresponding command-line arguments. Command-line arguments take precedence.

4. **Environment Structure**: The terrain, food positions, colonies, and walls are all specified in the environment file and cannot be overridden by command-line arguments.


## Creating Custom Environments

The simulation supports custom environments through text files with a specific format. To create your own environment, use the following sections and syntax:

### Environment File Format

Environment files are structured into sections, each with a specific purpose:

1. **DIMENSIONS**: (Required)

   ```plaintext
   DIMENSIONS:
   <width> <height>
   ```

   Defines the width and height of the environment grid in cells.

2. **COLONY**: (Required)

   ```plaintext
   COLONY:
   <x> <y>
   ```

   Specifies the coordinates of the ant colony (nest) location.

3. **FOOD**: (Required)

   ```plaintext
   FOOD:
   <x1> <y1> <amount1>
   <x2> <y2> <amount2>
   ...
   ```

   Defines food sources with their coordinates and amounts. Each line represents one food source.

4. **WALL**: (Optional)

   ```plaintext
   WALL:
   <x1> <y1>
   <x2> <y2>
   ...
   ```

   Specifies wall locations as individual cells. Each line represents one wall cell.

5. **ANTS**: (Optional)

   ```plaintext
   ANTS:
   <count>
   ```

   Specifies the number of ants to create. If present, this overrides the `--ants` command-line argument.

6. **TIME_LIMIT**: (Optional)

   ```plaintext
   TIME_LIMIT:
   <seconds>
   ```

   Specifies a time limit in seconds for the simulation (0 = no limit). Command-line argument `--time-limit` will override this if provided.

7. **MAX_STEPS**: (Optional)

   ```plaintext
   MAX_STEPS:
   <steps>
   ```

   Specifies maximum number of simulation steps (0 = unlimited). Command-line argument `--max-steps` will override this if provided.

### Example Environment File

Here's a simple example of an environment file:

```plaintext
DIMENSIONS:
100 100

# Central colony with food in corners
COLONY:
50 50

# Food sources in the corners
FOOD:
10 10 10
90 10 10
10 90 10
90 90 10

# Number of ants to spawn
ANTS:
50

# Time limit in seconds (0 = no limit)
TIME_LIMIT:
60

# Maximum simulation steps (0 = no limit)
MAX_STEPS:
10000
```

This creates a 100x100 environment with a colony in the center, food sources in each corner, 50 ants, a time limit of 60 seconds, and a maximum of 10000 steps.

### Comments and Formatting

- Lines starting with `#` are treated as comments and ignored
- Each section must begin with the section name followed by a colon
- Blank lines are ignored
- Coordinates are zero-indexed, with (0,0) at the top-left corner
