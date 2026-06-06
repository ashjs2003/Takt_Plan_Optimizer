import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def best_candidate(metrics_db):
    best_rank = min(candidate.get("pareto-rank", 1) for candidate in metrics_db)
    front = [candidate for candidate in metrics_db if candidate.get("pareto-rank", 1) == best_rank]
    metrics = np.array([candidate["metrics"] for candidate in front], dtype=float)
    metric_range = metrics.max(axis=0) - metrics.min(axis=0)
    normalized = np.divide(
        metrics - metrics.min(axis=0),
        metric_range,
        out=np.zeros_like(metrics),
        where=metric_range != 0,
    )
    return front[int(np.argmin(normalized.sum(axis=1)))]


def log_generation_best(generation, metrics_db, csv_path="./outputs/best_candidates.csv"):
    candidate = best_candidate(metrics_db)
    metrics = candidate["metrics"]
    row = {
        "generation": generation,
        "duration": metrics[0],
        "idle_time": metrics[1],
        "connectedness_penalty": metrics[2],
        "crew_cost": metrics[3],
        "pareto_rank": candidate.get("pareto-rank"),
        "crowding_distance": candidate.get("crowding-distance"),
        "overall_pareto_rank": candidate.get("overall-pareto-rank"),
        "zones": json.dumps(_json_ready(candidate["zones"])),
        "crews": json.dumps(_json_ready(candidate["crews"])),
    }

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def log_generation_population(generation, metrics_db, csv_path="./outputs/population_candidates.csv"):
    rows = []
    for candidate in metrics_db:
        metrics = candidate["metrics"]
        rows.append({
            "generation": generation,
            "duration": metrics[0],
            "idle_time": metrics[1],
            "connectedness_penalty": metrics[2],
            "crew_cost": metrics[3],
            "pareto_rank": candidate.get("pareto-rank"),
            "crowding_distance": candidate.get("crowding-distance"),
            "overall_pareto_rank": candidate.get("overall-pareto-rank"),
            "zones": json.dumps(_json_ready(candidate["zones"])),
            "crews": json.dumps(_json_ready(candidate["crews"])),
        })

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def plot_zoning(candidate, output_path, room_boundaries_path="./data/Room_Boundaries.csv"):
    room_boundaries = pd.read_csv(room_boundaries_path)
    room_boundaries = room_boundaries[room_boundaries["Level"].eq("L 1")]
    z = candidate["zones"]
    metrics = candidate["metrics"]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.get_cmap("viridis")
    zone_labels = sorted(set(z.values()))
    zone_color = {
        zone_id: colors(i / max(1, len(zone_labels) - 1))
        for i, zone_id in enumerate(zone_labels)
    }

    for room_id, room_df in room_boundaries.groupby("RoomId"):
        room_id = int(room_id)
        zone_id = z.get(room_id)
        if zone_id is None:
            continue

        min_x = min(room_df["Start X (ft)"].min(), room_df["End X (ft)"].min())
        max_x = max(room_df["Start X (ft)"].max(), room_df["End X (ft)"].max())
        min_y = min(room_df["Start Y (ft)"].min(), room_df["End Y (ft)"].min())
        max_y = max(room_df["Start Y (ft)"].max(), room_df["End Y (ft)"].max())
        ax.add_patch(Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            facecolor=zone_color[zone_id],
            edgecolor="none",
            alpha=0.5,
        ))

        for _, loop_df in room_df.groupby("Boundary Loop"):
            loop_df = loop_df.sort_values("Segment Index")
            ax.plot(loop_df["Start X (ft)"], loop_df["Start Y (ft)"], color="black", linewidth=0.7)

    for room_id, zone_id in z.items():
        room_rows = room_boundaries[room_boundaries["RoomId"].eq(int(room_id))]
        if room_rows.empty:
            continue
        x = room_rows["Room Location X (ft)"].iloc[0]
        y = room_rows["Room Location Y (ft)"].iloc[0]
        ax.text(
            x,
            y,
            str(zone_id),
            ha="center",
            va="center",
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8, "pad": 1.5},
            zorder=10,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"duration={metrics[0]:.2f}, idle={metrics[1]:.2f}, "
        f"connectedness={metrics[2]}, crew_cost={metrics[3]}"
    )
    ax.set_xlabel("X (ft)")
    ax.set_ylabel("Y (ft)")
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=min(zone_labels), vmax=max(zone_labels)), cmap=colors),
        ax=ax,
        label="Zone order",
    )
    colorbar.set_ticks(zone_labels)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_duration_crew(log_csv_path="./outputs/population_candidates.csv", output_path="./outputs/pareto_duration_crew.png", best_candidate=None):
    metrics_df = pd.read_csv(log_csv_path)
    idle_range = metrics_df["idle_time"].max() - metrics_df["idle_time"].min()
    if idle_range == 0:
        marker_sizes = 60
    else:
        marker_sizes = 40 + 160 * (metrics_df["idle_time"] - metrics_df["idle_time"].min()) / idle_range

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        metrics_df["crew_cost"],
        metrics_df["duration"],
        c=metrics_df["generation"],
        s=marker_sizes,
        cmap="viridis",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Crew Cost")
    ax.set_ylabel("Duration")
    ax.set_title("Pareto Plot: Duration vs Crew Cost")
    fig.colorbar(scatter, ax=ax, label="Generation")

    if best_candidate is None and "pareto_rank" in metrics_df.columns:
        final_generation = metrics_df["generation"].max()
        final_rows = metrics_df[metrics_df["generation"] == final_generation]
        best_row = best_generation_row(final_rows)
        best_metrics = [
            best_row["duration"],
            best_row["idle_time"],
            best_row["connectedness_penalty"],
            best_row["crew_cost"],
        ]
    elif best_candidate is not None:
        best_metrics = best_candidate["metrics"]
    else:
        best_metrics = None

    if best_metrics is not None:
        ax.scatter(
            best_metrics[3],
            best_metrics[0],
            marker="*",
            s=250,
            color="red",
            edgecolor="black",
            linewidth=0.8,
            label="Final best",
            zorder=5,
        )
        ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def best_generation_row(generation_df):
    front = generation_df[generation_df["pareto_rank"] == generation_df["pareto_rank"].min()]
    metric_columns = ["duration", "idle_time", "connectedness_penalty", "crew_cost"]
    metrics = front[metric_columns]
    metric_range = metrics.max() - metrics.min()
    normalized = metrics.subtract(metrics.min()).divide(metric_range.replace(0, 1))
    return front.loc[normalized.sum(axis=1).idxmin()]


def plot_metric_trends(log_csv_path="./outputs/population_candidates.csv", output_dir="./outputs"):
    metrics_df = pd.read_csv(log_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = ["duration", "idle_time", "connectedness_penalty", "crew_cost"]
    best_rows = pd.DataFrame([
        best_generation_row(generation_df)
        for _, generation_df in metrics_df.groupby("generation")
    ])

    for metric in metric_columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best_rows["generation"], best_rows[metric], marker="o")
        ax.set_xlabel("Generation")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Generation")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / f"{metric}_by_generation.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_takt_plan(candidate, output_path="./outputs/best_takt_plan.png"):
    from scheduler import make_schedule

    z = candidate["zones"]
    c = candidate["crews"]
    S, D = make_schedule(z, c)
    zones = sorted(set(z.values()))
    trades = list(c.keys())
    colors = plt.get_cmap("tab10")
    trade_colors = {trade: colors(i % 10) for i, trade in enumerate(trades)}

    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(zones))))

    for i, zone_id in enumerate(zones):
        for j, trade in enumerate(trades):
            if i > 0:
                previous_finish = S[i - 1, j] + D[i - 1, j]
                idle_duration = S[i, j] - previous_finish
                if idle_duration > 0:
                    ax.barh(
                        i,
                        idle_duration,
                        left=previous_finish,
                        height=0.18,
                        color=trade_colors[trade],
                        edgecolor=trade_colors[trade],
                        alpha=0.5,
                    )

            if D[i, j] > 0:
                ax.barh(
                    i,
                    D[i, j],
                    left=S[i, j],
                    height=0.6,
                    color=trade_colors[trade],
                    edgecolor="black",
                    linewidth=0.5,
                )

    metrics = candidate["metrics"]
    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels(zones)
    ax.set_xlabel("Time")
    ax.set_ylabel("Zone")
    ax.set_title(
        f"Takt Plan: duration={metrics[0]:.2f}, idle={metrics[1]:.2f}, "
        f"connectedness={metrics[2]}, crew_cost={metrics[3]}"
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor=trade_colors[trade], edgecolor="black", label=f"{trade} ({c[trade]} crews)")
            for trade in trades
        ],
        loc="lower right",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def candidate_from_log_row(row):
    return {
        "zones": {int(k): v for k, v in json.loads(row["zones"]).items()},
        "crews": json.loads(row["crews"]),
        "metrics": [
            row["duration"],
            row["idle_time"],
            row["connectedness_penalty"],
            row["crew_cost"],
        ],
    }
