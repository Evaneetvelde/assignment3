import argparse
import csv
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import SimulationRunner
from utils import add_ants, create_environment


DEFAULT_ENV = Path(__file__).parent / "envs" / "05_square_four_food_spots.txt"


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def regular_values(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def run_one(
    env_path: Path,
    ants: int,
    strategy: str,
    max_steps: int,
    time_limit: float,
    seed: int,
    evaporation_rate: float | None = None,
) -> dict:
    random.seed(seed)

    env = create_environment(str(env_path), width=100, height=100, verbose=False)
    env.requested_ant_count = ants

    if evaporation_rate is not None:
        env.home_pheromones.evaporation_rate = evaporation_rate
        env.food_pheromones.evaporation_rate = evaporation_rate

    add_ants(env, strategy, strategy_file=None, count=ants, verbose=False)

    runner = SimulationRunner(
        env,
        max_steps=max_steps,
        progress_interval=max_steps + 1,
        time_limit=time_limit,
    )
    return runner.run(verbose=False)


def summarize(rows: list[dict], group_key: str) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)

    summary = []
    for key, group in grouped.items():
        steps = [int(row["steps"]) for row in group]
        times = [float(row["time_taken"]) for row in group]
        successes = [1 if row["success"] else 0 for row in group]
        summary.append(
            {
                group_key: key,
                "runs": len(group),
                "steps_mean": mean(steps),
                "steps_std": stdev(steps) if len(steps) > 1 else 0.0,
                "time_mean": mean(times),
                "time_std": stdev(times) if len(times) > 1 else 0.0,
                "success_rate": mean(successes),
                "food_collected_mean": mean(float(row["food_collected"]) for row in group),
                "food_removed_mean": mean(float(row["food_removed"]) for row in group),
            }
        )

    return sorted(summary, key=lambda row: float(row[group_key]))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_errorbar(
    summary: list[dict],
    x_key: str,
    y_key: str,
    yerr_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    x = [float(row[x_key]) for row in summary]
    y = [float(row[y_key]) for row in summary]
    yerr = [float(row[yerr_key]) for row in summary]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def run_sweep(
    values: Iterable[float | int],
    raw_key: str,
    env_path: Path,
    runs: int,
    strategy: str,
    max_steps: int,
    time_limit: float,
    seed_base: int,
    configure: Callable[[float | int], dict],
) -> tuple[list[dict], list[dict]]:
    raw_rows = []
    for value in values:
        print(f"{raw_key}={value}")
        for run_index in range(runs):
            seed = seed_base + len(raw_rows)
            result = run_one(
                env_path=env_path,
                strategy=strategy,
                max_steps=max_steps,
                time_limit=time_limit,
                seed=seed,
                **configure(value),
            )
            raw_rows.append(
                {
                    raw_key: value,
                    "run": run_index + 1,
                    "seed": seed,
                    "success": result["success"],
                    "steps": result["steps"],
                    "time_taken": result["time_taken"],
                    "food_collected": result["food_collected"],
                    "food_removed": result["food_removed"],
                    "total_food": result["total_food"],
                }
            )

    return raw_rows, summarize(raw_rows, raw_key)


def copy_alias(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    source_png = source.with_suffix(".png")
    if source_png.exists():
        destination.with_suffix(".png").write_bytes(source_png.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate assignment Q1/Q2 sweep plots and CSV data."
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--out-dir", type=Path, default=Path("report_template") / "figures")
    parser.add_argument("--data-dir", type=Path, default=Path("results"))
    parser.add_argument("--strategy", default="smart")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--questions",
        default="q1,q2",
        help="Comma-separated subset: q1,q2",
    )
    parser.add_argument(
        "--ant-counts",
        default="1,20,40,60,80,100,120,140,160,180,200",
        help="Comma-separated values for Q1.",
    )
    parser.add_argument("--evaporation-points", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--no-template-aliases",
        action="store_true",
        help="Do not write the filenames used directly in the provided Report.tex comments.",
    )
    args = parser.parse_args()

    questions = {part.strip().lower() for part in args.questions.split(",")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if "q1" in questions:
        ant_counts = parse_int_list(args.ant_counts)
        raw, summary = run_sweep(
            values=ant_counts,
            raw_key="ants",
            env_path=args.env,
            runs=args.runs,
            strategy=args.strategy,
            max_steps=args.max_steps,
            time_limit=args.time_limit,
            seed_base=args.seed,
            configure=lambda ants: {"ants": int(ants)},
        )
        write_csv(args.data_dir / "q1_ant_count_raw.csv", raw)
        write_csv(args.data_dir / "q1_ant_count_summary.csv", summary)

        steps_path = args.out_dir / "q1_ant_count_steps.pdf"
        time_path = args.out_dir / "q1_ant_count_time.pdf"
        plot_errorbar(
            summary,
            x_key="ants",
            y_key="steps_mean",
            yerr_key="steps_std",
            xlabel="Number of ants",
            ylabel="Steps to objective",
            title="Q1 - Steps vs number of ants",
            path=steps_path,
        )
        plot_errorbar(
            summary,
            x_key="ants",
            y_key="time_mean",
            yerr_key="time_std",
            xlabel="Number of ants",
            ylabel="Runtime (seconds)",
            title="Q1 - Runtime vs number of ants",
            path=time_path,
        )
        if not args.no_template_aliases:
            copy_alias(steps_path, args.out_dir / "q4_ant_count_steps.pdf")
            copy_alias(time_path, args.out_dir / "q4_ant_count_time.pdf")

    if "q2" in questions:
        evaporation_rates = regular_values(0.500, 0.999, args.evaporation_points)
        raw, summary = run_sweep(
            values=evaporation_rates,
            raw_key="evaporation_rate",
            env_path=args.env,
            runs=args.runs,
            strategy=args.strategy,
            max_steps=args.max_steps,
            time_limit=args.time_limit,
            seed_base=args.seed + 100000,
            configure=lambda rate: {"ants": 70, "evaporation_rate": float(rate)},
        )
        write_csv(args.data_dir / "q2_evaporation_raw.csv", raw)
        write_csv(args.data_dir / "q2_evaporation_summary.csv", summary)

        evaporation_path = args.out_dir / "q2_evaporation.pdf"
        plot_errorbar(
            summary,
            x_key="evaporation_rate",
            y_key="steps_mean",
            yerr_key="steps_std",
            xlabel="Pheromone evaporation rate",
            ylabel="Steps to objective",
            title="Q2 - Steps vs pheromone evaporation rate",
            path=evaporation_path,
        )
        if not args.no_template_aliases:
            copy_alias(evaporation_path, args.out_dir / "q3_evaporation.pdf")

    print(f"Done. Figures: {args.out_dir}")
    print(f"Done. CSV data: {args.data_dir}")


if __name__ == "__main__":
    main()
