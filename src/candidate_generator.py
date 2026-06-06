import pandas as pd
import numpy as np

def zone_grouping_initializer(num_zones: int):
    quantities_db = pd.read_csv("./data/quantities_db.csv")
    productivity_db = pd.read_csv("./data/Takt_Productivity_Rates.csv")
    room_boundaries_db = pd.read_csv("./data/Room_Boundaries.csv")
    room_boundaries_db = room_boundaries_db[room_boundaries_db["Level"].eq("L 1")]

    room_ids = sorted(map(int, quantities_db["room_id"].unique()))

    # workload based
    w = {room_id: 0.0 for room_id in room_ids}
    for room_id in room_ids:
        for _, trade_row in productivity_db.iterrows():
            quantity = sum(quantities_db[
                (quantities_db["room_id"] == room_id) &
                (quantities_db["trade"] == trade_row["trade"])
            ]["quantity"], 0.0)
            w[room_id] += quantity / trade_row["rate"]

    target_workload = sum(w.values()) / num_zones

    room_locations = room_boundaries_db.groupby("RoomId").first()
    x = room_locations["Room Location X (ft)"]
    y = room_locations["Room Location Y (ft)"]
    coords = {room_id: np.array([x.loc[room_id], y.loc[room_id]]) for room_id in room_ids}

    # choose "num_zones" rooms through farthest point sampling: where the rooms farthest away from each other are chosen
    seeds = [max(room_ids, key=lambda room_id: w[room_id])]
    while len(seeds) < num_zones:
        seeds.append(max(
            [room_id for room_id in room_ids if room_id not in seeds],
            key=lambda room_id: min(np.linalg.norm(coords[room_id] - coords[seed]) for seed in seeds)
        ))

    # build room adjacency 
    adjacency = {room_id: set() for room_id in room_ids}
    for _, boundary_df in room_boundaries_db.dropna(subset=["Boundary Element Id"]).groupby("Boundary Element Id"):
        adjacent_rooms = set(map(int, boundary_df["RoomId"].unique()))
        for room_id in adjacent_rooms:
            adjacency[room_id].update(adjacent_rooms - {room_id})

    # initialize
    zones = {j: {seed} for j, seed in enumerate(seeds)}
    zone_workloads = {j: w[seed] for j, seed in enumerate(seeds)}
    z = {seed: j for j, seed in enumerate(seeds)}
    unassigned = set(room_ids) - set(seeds)

    alpha = 1.0
    gamma = 1.0/ 130 # normalized by floor length
    eta = 1.0

    # zone growth loop
    while unassigned:
        candidates = []
        for j in zones:
            # consider all unassigned rooms that are adjacent to the current zone as candidates for assignment to the zone
            zone_candidates = set().union(*(adjacency[room_id] for room_id in zones[j])) & unassigned
            for room_id in zone_candidates:
                centroid = np.mean([coords[zone_room] for zone_room in zones[j]], axis=0)
                
                workload_delta = ((zone_workloads[j] + w[room_id] - target_workload) ** 2 -
                                  (zone_workloads[j] - target_workload) ** 2)
                distance_cost = np.linalg.norm(coords[room_id] - centroid) ** 2
                adjacency_support = len(adjacency[room_id] & zones[j])
                
                # incremental cost function
                cost = alpha * workload_delta + gamma * distance_cost - eta * adjacency_support
                
                candidates.append((cost, room_id, j))

        # if no adjacent candidates, assign the unassigned room with the least workload to the zone with the least workload
        if not candidates:
            candidates = [(0.0, room_id, min(zones, key=lambda j: zone_workloads[j])) for room_id in unassigned]

        # update the zone assignment
        _, room_id, j = min(candidates)
        zones[j].add(room_id)
        zone_workloads[j] += w[room_id]
        z[room_id] = j
        unassigned.remove(room_id)

    return z

def zone_ordering_initializer(z):
    room_boundaries_db = pd.read_csv("./data/Room_Boundaries.csv")
    room_boundaries_db = room_boundaries_db[room_boundaries_db["Level"].eq("L 1")]
    room_locations = room_boundaries_db.groupby("RoomId").first()
    coords = {
        int(room_id): np.array([
            row["Room Location X (ft)"],
            row["Room Location Y (ft)"],
        ])
        for room_id, row in room_locations.iterrows()
    }

    zones = {}
    for room_id, zone_id in z.items():
        zones.setdefault(zone_id, []).append(room_id)

    sweep_vectors = [
        np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        for angle in range(0, 360, 45)
    ]

    ordered_z = []
    for sweep_vector in sweep_vectors:
        zone_scores = {}
        for zone_id, room_ids in zones.items():
            centroid = np.mean([coords[room_id] for room_id in room_ids], axis=0)
            zone_scores[zone_id] = centroid @ sweep_vector

        ordered_zone_ids = sorted(zone_scores, key=zone_scores.get)
        ordered_labels = {zone_id: i + 1 for i, zone_id in enumerate(ordered_zone_ids)}
        ordered_z.append({room_id: ordered_labels[zone_id] for room_id, zone_id in z.items()})

    return ordered_z

def crew_initializer():
    crew_candidates = []
    crew_data = pd.read_csv("./data/Takt_Productivity_Rates.csv")

    max_candidate = {}
    min_candidate = {}
    mid_candidate = {}

    for _, row in crew_data.iterrows():
        max_candidate[row["trade"]] = row["max_available"]
        min_candidate[row["trade"]] = 1
        mid_candidate[row["trade"]] = row["max_available"] // 2

    crew_candidates.append(max_candidate)
    crew_candidates.append(min_candidate)
    crew_candidates.append(mid_candidate)
    
    return crew_candidates
