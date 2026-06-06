import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from candidate_generator import zone_grouping_initializer, zone_ordering_initializer, crew_initializer
from logger import best_candidate, candidate_from_log_row, log_generation_best, log_generation_population, plot_metric_trends, plot_pareto_duration_crew, plot_takt_plan, plot_zoning
from scheduler import make_schedule, evaluate_schedule

def dominates(candidate_a, candidate_b):
    a_metrics = candidate_a["metrics"]
    b_metrics = candidate_b["metrics"]
    return all(a <= b for a, b in zip(a_metrics, b_metrics)) and any(a < b for a, b in zip(a_metrics, b_metrics))

def add_crowding_distance(metrics_db, front):
    for i in front:
        metrics_db[i]["crowding-distance"] = 0.0

    num_objectives = len(metrics_db[front[0]]["metrics"])
    for objective in range(num_objectives):
        front.sort(key=lambda i: metrics_db[i]["metrics"][objective])
        metrics_db[front[0]]["crowding-distance"] = np.inf
        metrics_db[front[-1]]["crowding-distance"] = np.inf

        objective_min = metrics_db[front[0]]["metrics"][objective]
        objective_max = metrics_db[front[-1]]["metrics"][objective]
        if objective_max == objective_min:
            continue

        for k in range(1, len(front) - 1):
            previous_metric = metrics_db[front[k - 1]]["metrics"][objective]
            next_metric = metrics_db[front[k + 1]]["metrics"][objective]
            metrics_db[front[k]]["crowding-distance"] += (next_metric - previous_metric) / (objective_max - objective_min)

def add_pareto_ranks(metrics_db):
    remaining = set(range(len(metrics_db)))
    rank = 1

    while remaining:
        front = []
        for i in remaining:
            if not any(dominates(metrics_db[j], metrics_db[i]) for j in remaining if j != i):
                front.append(i)

        for i in front:
            metrics_db[i]["pareto-rank"] = rank

        add_crowding_distance(metrics_db, front)
        remaining -= set(front)
        rank += 1

    sorted_candidates = sorted(
        range(len(metrics_db)),
        key=lambda i: (metrics_db[i]["pareto-rank"], -metrics_db[i]["crowding-distance"])
    )
    for overall_rank, i in enumerate(sorted_candidates, start=1):
        metrics_db[i]["overall-pareto-rank"] = overall_rank

    return metrics_db

def tournament_selection(metrics_db):
    parents = []
    for _ in range(2):
        i, j = np.random.choice(len(metrics_db), size=2, replace=False)
        candidate_a = metrics_db[i]
        candidate_b = metrics_db[j]
        candidates = [candidate_a, candidate_b]
        parents.append(min(candidates, key=lambda c: c["overall-pareto-rank"]))
    return parents

def group_zones(z):
    # group rooms by zones
    zones = {}
    for room_id, zone_id in z.items():
        zones.setdefault(zone_id, set()).add(room_id)
    return zones

def crossover_z(parent_a, parent_b, strategy="current", inheritance_ratio=0.6):
    parents_by_role = {
        "connectedness": parent_a if parent_a["metrics"][2] <= parent_b["metrics"][2] else parent_b,
        "idle": parent_a if parent_a["metrics"][1] <= parent_b["metrics"][1] else parent_b,
        "duration": parent_a if parent_a["metrics"][0] <= parent_b["metrics"][0] else parent_b,
    }

    if isinstance(strategy, tuple):
        block_parent = parents_by_role[strategy[0]]
        fill_parent = parents_by_role[strategy[1]]
        order_parent = parents_by_role[strategy[2]]
    elif strategy == "connectedness_first":
        block_parent = parents_by_role["connectedness"]
        fill_parent = parents_by_role["idle"]
        order_parent = parents_by_role["duration"]
    elif strategy == "idle_first" or strategy == "current":
        block_parent = parents_by_role["idle"]
        fill_parent = parents_by_role["connectedness"]
        order_parent = parents_by_role["duration"]
    elif strategy == "random_blocks":
        block_parent = np.random.choice([parent_a, parent_b])
        fill_parent = parent_b if block_parent is parent_a else parent_a
        order_parent = np.random.choice([parent_a, parent_b])
    else:
        block_parent = parents_by_role["duration"]
        fill_parent = parents_by_role["connectedness"]
        order_parent = parents_by_role["idle"]

    block_zones = group_zones(block_parent["zones"])
    fill_zones = group_zones(fill_parent["zones"])
    target_num_zones = len(set(order_parent["zones"].values()))
    num_block_zones = min(len(block_zones), max(1, int(np.ceil(inheritance_ratio * target_num_zones))))
    selected_block_zones = np.random.choice(list(block_zones.keys()), size=num_block_zones, replace=False)

    child_zones = []
    assigned_rooms = set()
    for zone_id in selected_block_zones:
        rooms = set(block_zones[zone_id])
        child_zones.append(rooms)
        assigned_rooms.update(rooms)

    for zone_id in fill_zones:
        rooms = fill_zones[zone_id] - assigned_rooms
        if rooms:
            child_zones.append(rooms)
            assigned_rooms.update(rooms)

    child_zones.sort(key=lambda rooms: np.mean([order_parent["zones"][room_id] for room_id in rooms]))

    child_z = {}
    for zone_id, rooms in enumerate(child_zones, start=1):
        for room_id in rooms:
            child_z[room_id] = zone_id

    return child_z

def crossover_c(parent_a, parent_b):
    productivity_db = pd.read_csv("./data/Takt_Productivity_Rates.csv")
    max_crews = dict(zip(productivity_db["trade"], productivity_db["max_available"]))

    child_c = {}
    for trade in max_crews:
        crew_count = parent_a["crews"][trade] if np.random.random() < 0.5 else parent_b["crews"][trade]
        child_c[trade] = min(max(1, crew_count), max_crews[trade])

    return child_c

def is_valid_candidate(z, c):
    room_boundaries_db = pd.read_csv("./data/Room_Boundaries.csv")
    room_boundaries_db = room_boundaries_db[room_boundaries_db["Level"].eq("L 1")]
    productivity_db = pd.read_csv("./data/Takt_Productivity_Rates.csv")

    # Every room appears exactly once.
    expected_rooms = set(map(int, room_boundaries_db["RoomId"].unique()))
    candidate_rooms = set(map(int, z.keys()))
    if candidate_rooms != expected_rooms:
        return False

    # Labels are contiguous: 1,...,p.
    zone_labels = sorted(set(z.values()))
    if zone_labels != list(range(1, len(zone_labels) + 1)):
        return False

    # No empty zones.
    if any(len(rooms) == 0 for rooms in group_zones(z).values()):
        return False

    # Crew counts are positive integers and do not exceed max available.
    max_crews = dict(zip(productivity_db["trade"], productivity_db["max_available"]))
    for trade, max_available in max_crews.items():
        if trade not in c or c[trade] < 1 or c[trade] > max_available:
            return False

    return True

def relabel_zones(z):
    labels = {old_label: new_label for new_label, old_label in enumerate(sorted(set(z.values())), start=1)}
    return {room_id: labels[zone_id] for room_id, zone_id in z.items()}

def get_room_adjacency():
    room_boundaries_db = pd.read_csv("./data/Room_Boundaries.csv")
    room_boundaries_db = room_boundaries_db[room_boundaries_db["Level"].eq("L 1")]
    room_ids = set(map(int, room_boundaries_db["RoomId"].unique()))
    adjacency = {room_id: set() for room_id in room_ids}
    for _, boundary_df in room_boundaries_db.dropna(subset=["Boundary Element Id"]).groupby("Boundary Element Id"):
        adjacent_rooms = set(map(int, boundary_df["RoomId"].unique()))
        for room_id in adjacent_rooms:
            adjacency[room_id].update(adjacent_rooms - {room_id})
    return adjacency

def get_zone_workloads(z):
    quantities_db = pd.read_csv("./data/quantities_db.csv")
    productivity_db = pd.read_csv("./data/Takt_Productivity_Rates.csv")

    room_workloads = {room_id: 0.0 for room_id in z}
    for room_id in room_workloads:
        for _, trade_row in productivity_db.iterrows():
            quantity = sum(quantities_db[
                (quantities_db["room_id"] == room_id) &
                (quantities_db["trade"] == trade_row["trade"])
            ]["quantity"], 0.0)
            room_workloads[room_id] += quantity / trade_row["rate"]

    zone_workloads = {zone_id: 0.0 for zone_id in set(z.values())}
    for room_id, zone_id in z.items():
        zone_workloads[zone_id] += room_workloads[room_id]
    return zone_workloads

def mutate_boundary_room_move(z):
    z = z.copy()
    adjacency = get_room_adjacency()
    zone_workloads = get_zone_workloads(z)
    overloaded_zone = max(zone_workloads, key=zone_workloads.get)

    movable_rooms = [
        room_id for room_id, zone_id in z.items()
        if zone_id == overloaded_zone and any(z[neighbor] != zone_id for neighbor in adjacency[room_id] if neighbor in z)
    ]
    if movable_rooms:
        room_id = np.random.choice(movable_rooms)
        neighbor_zones = [z[neighbor] for neighbor in adjacency[room_id] if neighbor in z and z[neighbor] != overloaded_zone]
        z[room_id] = min(neighbor_zones, key=lambda zone_id: zone_workloads.get(zone_id, 0.0))
    return relabel_zones(z)

def mutate_boundary_room_swap(z):
    z = z.copy()
    adjacency = get_room_adjacency()
    boundary_pairs = [
        (room_id, neighbor)
        for room_id in z
        for neighbor in adjacency[room_id]
        if neighbor in z and z[room_id] != z[neighbor]
    ]
    if boundary_pairs:
        room_a, room_b = boundary_pairs[np.random.randint(len(boundary_pairs))]
        z[room_a], z[room_b] = z[room_b], z[room_a]
    return relabel_zones(z)

def mutate_split_overloaded_zone(z):
    z = z.copy()
    zones = group_zones(z)
    zone_workloads = get_zone_workloads(z)
    zone_id = max(zone_workloads, key=zone_workloads.get)
    rooms = list(zones[zone_id])
    if len(rooms) > 1:
        np.random.shuffle(rooms)
        new_zone_id = max(zones) + 1
        for room_id in rooms[len(rooms) // 2:]:
            z[room_id] = new_zone_id
    return relabel_zones(z)

def mutate_merge_underloaded_zones(z):
    z = z.copy()
    adjacency = get_room_adjacency()
    zone_workloads = get_zone_workloads(z)
    zone_pairs = {
        tuple(sorted((z[room_id], z[neighbor])))
        for room_id in z
        for neighbor in adjacency[room_id]
        if neighbor in z and z[room_id] != z[neighbor]
    }
    if zone_pairs:
        zone_a, zone_b = min(zone_pairs, key=lambda pair: zone_workloads[pair[0]] + zone_workloads[pair[1]])
        for room_id, zone_id in z.items():
            if zone_id == zone_b:
                z[room_id] = zone_a
    return relabel_zones(z)

def mutate_zone_order_swap(z):
    z = z.copy()
    labels = sorted(set(z.values()))
    if len(labels) >= 2:
        zone_a, zone_b = np.random.choice(labels, size=2, replace=False)
        for room_id, zone_id in z.items():
            if zone_id == zone_a:
                z[room_id] = zone_b
            elif zone_id == zone_b:
                z[room_id] = zone_a
    return relabel_zones(z)

def mutate_zone_order_insert(z):
    labels = sorted(set(z.values()))
    if len(labels) < 2:
        return z.copy()
    old_index, new_index = np.random.choice(range(len(labels)), size=2, replace=False)
    moved_label = labels.pop(old_index)
    labels.insert(new_index, moved_label)
    label_map = {old_label: new_label for new_label, old_label in enumerate(labels, start=1)}
    return {room_id: label_map[zone_id] for room_id, zone_id in z.items()}

def mutate_reverse_zone_subsequence(z):
    labels = sorted(set(z.values()))
    if len(labels) < 2:
        return z.copy()
    i, j = sorted(np.random.choice(range(len(labels)), size=2, replace=False))
    labels[i:j + 1] = reversed(labels[i:j + 1])
    label_map = {old_label: new_label for new_label, old_label in enumerate(labels, start=1)}
    return {room_id: label_map[zone_id] for room_id, zone_id in z.items()}

def mutate_crew_count(c, direction=None):
    c = c.copy()
    productivity_db = pd.read_csv("./data/Takt_Productivity_Rates.csv")
    max_crews = dict(zip(productivity_db["trade"], productivity_db["max_available"]))

    if direction == "add":
        trades = [trade for trade in c if c[trade] < max_crews[trade]]
        if trades:
            c[np.random.choice(trades)] += 1
    elif direction == "remove":
        trades = [trade for trade in c if c[trade] > 1]
        if trades:
            c[np.random.choice(trades)] -= 1
    else:
        trade = np.random.choice(list(c.keys()))
        delta = np.random.choice([-1, 1])
        c[trade] = min(max(1, c[trade] + delta), max_crews[trade])

    return c

def mutation_objective(metrics, metrics_db):
    population_metrics = np.array([candidate["metrics"] for candidate in metrics_db], dtype=float)
    metric_min = population_metrics.min(axis=0)
    metric_max = population_metrics.max(axis=0)
    metric_range = metric_max - metric_min
    normalized_metrics = np.divide(
        np.array(metrics, dtype=float) - metric_min,
        metric_range,
        out=np.zeros(len(metrics), dtype=float),
        where=metric_range != 0,
    )

    if normalized_metrics.sum() == 0:
        return np.random.choice(range(len(metrics)))
    return np.random.choice(range(len(metrics)), p=normalized_metrics / normalized_metrics.sum())

def mutate_candidate(z, c, metrics, metrics_db, disabled_mutations=None):
    z = z.copy()
    c = c.copy()
    disabled_mutations = set(disabled_mutations or [])
    objective = mutation_objective(metrics, metrics_db)

    if objective in {0, 2} and "zone" in disabled_mutations:
        objective = np.random.choice([1, 3])
    if objective == 1 and "order" in disabled_mutations:
        objective = np.random.choice([0, 2, 3])
    if objective == 3 and "crew" in disabled_mutations:
        objective = np.random.choice([0, 1, 2])

    if objective == 0:
        if "zone" not in disabled_mutations and np.random.random() < 0.5:
            z = mutate_split_overloaded_zone(z)
        else:
            if "crew" not in disabled_mutations:
                c = mutate_crew_count(c, direction="add")
    elif objective == 1:
        z = np.random.choice([
            mutate_zone_order_swap,
            mutate_zone_order_insert,
            mutate_reverse_zone_subsequence,
        ])(z)
    elif objective == 2:
        if np.random.random() < 0.5:
            z = mutate_boundary_room_move(z)
        else:
            z = mutate_boundary_room_swap(z)
    else:
        if "zone" not in disabled_mutations and np.random.random() < 0.5:
            z = mutate_merge_underloaded_zones(z)
        else:
            if "crew" not in disabled_mutations:
                c = mutate_crew_count(c, direction="remove")

    return z, c

def should_stop(generation, max_generations, hypervolume_history, k, epsilon, evaluations=None, max_evaluations=None):
    if generation >= max_generations:
        return True
    if max_evaluations is not None and evaluations is not None and evaluations >= max_evaluations:
        return True
    if len(hypervolume_history) <= k:
        return False

    hv_t = hypervolume_history[-1]
    hv_t_minus_k = hypervolume_history[-k - 1]
    if hv_t_minus_k == 0:
        return abs(hv_t - hv_t_minus_k) < epsilon

    return abs(hv_t - hv_t_minus_k) / abs(hv_t_minus_k) < epsilon

def evaluate_candidate(z, c):
    S, D = make_schedule(z, c)
    return {
        "zones": z,
        "crews": c,
        "metrics": evaluate_schedule(z, c, S, D),
    }

def estimate_hypervolume(metrics_db):
    front = [candidate for candidate in metrics_db if candidate["pareto-rank"] == 1]
    metrics = np.array([candidate["metrics"] for candidate in metrics_db], dtype=float)
    metric_min = metrics.min(axis=0)
    metric_max = metrics.max(axis=0)
    metric_range = metric_max - metric_min

    hypervolume = 0.0
    for candidate in front:
        normalized = np.divide(
            np.array(candidate["metrics"], dtype=float) - metric_min,
            metric_range,
            out=np.zeros(len(candidate["metrics"]), dtype=float),
            where=metric_range != 0,
        )
        hypervolume += np.prod(1 - normalized)
    return hypervolume

def initial_population(population_size):
    room_boundaries_db = pd.read_csv("./data/Room_Boundaries.csv")
    room_boundaries_db = room_boundaries_db[room_boundaries_db["Level"].eq("L 1")]
    num_rooms = room_boundaries_db["RoomId"].nunique()

    z_candidates = []
    for candidate_num_zones in range(1, num_rooms + 1):
        grouping = zone_grouping_initializer(candidate_num_zones)
        z_candidates.extend(zone_ordering_initializer(grouping))

    c_candidates = crew_initializer()
    metrics_db = []
    initial_candidates = [(z, c) for z in z_candidates for c in c_candidates]
    np.random.shuffle(initial_candidates)
    for z, c in initial_candidates:
        if is_valid_candidate(z, c):
            metrics_db.append(evaluate_candidate(z, c))
        if len(metrics_db) >= population_size:
            break

    return metrics_db

def run_nsga2(config=None):
    config = config or {}
    seed = config.get("seed")
    if seed is not None:
        np.random.seed(seed)

    population_size = config.get("population_size", 20)
    max_generations = config.get("max_generations", 20)
    max_evaluations = config.get("max_evaluations", 400)
    hypervolume_window = config.get("hypervolume_window", 5)
    epsilon = config.get("epsilon", 0.001)
    crossover_strategy = config.get("crossover_strategy", "current")
    inheritance_ratio = config.get("inheritance_ratio", 0.6)
    disabled_mutations = config.get("disabled_mutations", [])
    log_outputs = config.get("log_outputs", False)
    verbose = config.get("verbose", False)

    if log_outputs:
        Path("./outputs/best_candidates.csv").unlink(missing_ok=True)
        Path("./outputs/population_candidates.csv").unlink(missing_ok=True)

    metrics_db = initial_population(population_size)
    metrics_db = add_pareto_ranks(metrics_db)
    metrics_db = sorted(metrics_db, key=lambda candidate: candidate["overall-pareto-rank"])[:population_size]
    hypervolume_history = [estimate_hypervolume(metrics_db)]
    generation = 0
    evaluations = len(metrics_db)
    invalid_children = 0
    total_children = 0

    while not should_stop(
        generation,
        max_generations,
        hypervolume_history,
        hypervolume_window,
        epsilon,
        evaluations,
        max_evaluations,
    ):
        children = []
        for _ in range(population_size):
            parents = tournament_selection(metrics_db)

            child_z = crossover_z(
                parents[0],
                parents[1],
                strategy=crossover_strategy,
                inheritance_ratio=inheritance_ratio,
            )
            child_c = crossover_c(parents[0], parents[1])

            child_z, child_c = mutate_candidate(
                child_z,
                child_c,
                parents[0]["metrics"],
                metrics_db,
                disabled_mutations=disabled_mutations,
            )

            total_children += 1
            if is_valid_candidate(child_z, child_c):
                children.append(evaluate_candidate(child_z, child_c))
            else:
                invalid_children += 1

        metrics_db.extend(children)
        metrics_db = add_pareto_ranks(metrics_db)
        metrics_db = sorted(metrics_db, key=lambda candidate: candidate["overall-pareto-rank"])[:population_size]

        generation += 1
        evaluations += len(children)
        hypervolume_history.append(estimate_hypervolume(metrics_db))
        if log_outputs:
            log_generation_best(generation, metrics_db)
            log_generation_population(generation, metrics_db)
        best = best_candidate(metrics_db)
        best_metrics = best["metrics"]
        if verbose:
            timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"{timestamp}, generation={generation}, duration={best_metrics[0]}, idle={best_metrics[1]}, connectedness={best_metrics[2]}, crew_cost={best_metrics[3]}")

    pareto_front = [candidate for candidate in metrics_db if candidate["pareto-rank"] == 1]
    best = best_candidate(metrics_db)
    return {
        "metrics_db": metrics_db,
        "best": best,
        "hypervolume": hypervolume_history[-1],
        "num_pareto_solutions": len(pareto_front),
        "invalid_child_rate": invalid_children / total_children if total_children else 0.0,
        "generations": generation,
        "evaluations": evaluations,
    }

if __name__ == "__main__":
    #best config
    result = run_nsga2({
        "crossover_strategy": ("connectedness", "duration", "idle"),
        "log_outputs": True,
        "verbose": True,
    })
    best = result["best"]
    plot_zoning(best, "./outputs/best_zoning.png")
    plot_takt_plan(best)
    best_log = pd.read_csv("./outputs/best_candidates.csv")
    plot_takt_plan(candidate_from_log_row(best_log.iloc[0]), "./outputs/first_generation_takt_plan.png")
    plot_pareto_duration_crew(best_candidate=best)
    plot_metric_trends()
    
