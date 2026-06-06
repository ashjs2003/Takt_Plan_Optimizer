import time
from pathlib import Path

import pandas as pd

from baseline import baseline_candidate
from nsga2 import run_nsga2


BASE_CONFIG = {
    "population_size": 12,
    "max_generations": 8,
    "max_evaluations": 120,
    "hypervolume_window": 4,
    "epsilon": 0.001,
    "log_outputs": False,
    "verbose": False,
}


def experiment_configs():
    configs = [
        ("baseline", "zigzag_2_rooms", None),
        ("mutation_ablation", "full", {}),
        ("mutation_ablation", "no_zone_mutation", {"disabled_mutations": ["zone"]}),
        ("mutation_ablation", "no_order_mutation", {"disabled_mutations": ["order"]}),
        ("mutation_ablation", "no_crew_mutation", {"disabled_mutations": ["crew"]}),
        ("crossover_ablation", "current", {"crossover_strategy": "current"}),
        ("crossover_ablation", "connectedness_first", {"crossover_strategy": "connectedness_first"}),
        ("crossover_ablation", "duration_first", {"crossover_strategy": "duration_first"}),
        ("crossover_ablation", "block_connectedness_fill_duration_order_idle", {"crossover_strategy": ("connectedness", "duration", "idle")}),
        ("crossover_ablation", "block_idle_fill_duration_order_connectedness", {"crossover_strategy": ("idle", "duration", "connectedness")}),
        ("crossover_ablation", "block_duration_fill_idle_order_connectedness", {"crossover_strategy": ("duration", "idle", "connectedness")}),
        ("crossover_ablation", "random_blocks", {"crossover_strategy": "random_blocks"}),
        ("parameter_sensitivity", "pop_10", {"population_size": 10}),
        ("parameter_sensitivity", "pop_20", {"population_size": 20}),
        ("parameter_sensitivity", "inherit_40", {"inheritance_ratio": 0.4}),
        ("parameter_sensitivity", "inherit_80", {"inheritance_ratio": 0.8}),
    ]
    return configs


def run_baseline(experiment_name, variant, seed):
    start_time = time.time()
    candidate = baseline_candidate()
    metrics = candidate["metrics"]
    return {
        "experiment_name": experiment_name,
        "variant": variant,
        "seed": seed,
        "duration": metrics[0],
        "idle_time": metrics[1],
        "connectedness_penalty": metrics[2],
        "crew_cost": metrics[3],
        "hypervolume": None,
        "num_pareto_solutions": None,
        "invalid_child_rate": None,
        "generations": 0,
        "evaluations": 1,
        "runtime_seconds": time.time() - start_time,
    }


def run_optimizer(experiment_name, variant, overrides, seed):
    config = BASE_CONFIG | overrides | {"seed": seed}
    start_time = time.time()
    result = run_nsga2(config)
    metrics = result["best"]["metrics"]
    return {
        "experiment_name": experiment_name,
        "variant": variant,
        "seed": seed,
        "duration": metrics[0],
        "idle_time": metrics[1],
        "connectedness_penalty": metrics[2],
        "crew_cost": metrics[3],
        "hypervolume": result["hypervolume"],
        "num_pareto_solutions": result["num_pareto_solutions"],
        "invalid_child_rate": result["invalid_child_rate"],
        "generations": result["generations"],
        "evaluations": result["evaluations"],
        "runtime_seconds": time.time() - start_time,
    }


def summarize_results(runs_df):
    metric_columns = [
        "duration",
        "idle_time",
        "connectedness_penalty",
        "crew_cost",
        "hypervolume",
        "num_pareto_solutions",
        "invalid_child_rate",
        "runtime_seconds",
    ]
    summary = runs_df.groupby(["experiment_name", "variant"])[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(column).strip("_")
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


if __name__ == "__main__":
    output_dir = Path("./outputs/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = output_dir / "experiment_runs.csv"
    summary_path = output_dir / "experiment_summary.csv"
    existing_runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    completed = set()
    if not existing_runs.empty:
        completed = set(zip(
            existing_runs["experiment_name"],
            existing_runs["variant"],
            existing_runs["seed"].astype(int),
        ))

    seeds = list(range(3, 10))
    rows = []
    for experiment_name, variant, overrides in experiment_configs():
        if overrides is None:
            if existing_runs.empty:
                print(f"Running {experiment_name} / {variant}")
                rows.append(run_baseline(experiment_name, variant, seed=0))
            continue

        for seed in seeds:
            if (experiment_name, variant, seed) in completed:
                print(f"Skipping {experiment_name} / {variant} / seed={seed}")
                continue
            print(f"Running {experiment_name} / {variant} / seed={seed}")
            rows.append(run_optimizer(experiment_name, variant, overrides, seed))

    runs_df = pd.concat([existing_runs, pd.DataFrame(rows)], ignore_index=True)
    summary_df = summarize_results(runs_df)
    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
