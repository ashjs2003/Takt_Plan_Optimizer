import math
import pandas as pd
import numpy as np

def make_schedule(z, c):
    """
        z: ordered room zone vector
        c: crew num  per trade vector
    """
    zones = sorted(set(z.values()))
    trades = list(c.keys())
    num_zones = len(zones)
    num_trades = len(trades)

    quatntities_db = pd.read_csv("./data/quantities_db.csv")
    productivity_db = pd.read_csv("./data/Takt_Productivity_Rates.csv")

    # define Q: quantities per zone per trade
    Q = np.zeros((num_zones, num_trades))
    for i, zone_id in enumerate(zones):
        room_ids = [room_id for room_id, room_zone in z.items() if room_zone == zone_id]
        for j, trade in enumerate(trades):
            Q[i, j] = sum(quatntities_db[
                (quatntities_db["room_id"].isin(room_ids)) &
                (quatntities_db["trade"] == trade)
            ]["quantity"], 0.0)

    # define D: duration per zone per trade
    D = np.zeros((num_zones, num_trades))
    for j, trade in enumerate(trades):
        p = productivity_db[productivity_db["trade"] == trade]["rate"].values[0]
        for i in range(num_zones):
            D[i, j] = Q[i, j] / (p * c[trade])

    # define S: start time per zone per trade 
    S = np.zeros((num_zones, num_trades))
    for i in range(num_zones):
        for j in range(num_trades):

            # finish time of previous trade in the same zone
            prev_trade_finish = S[i, j-1] + D[i, j-1] if j > 0 else 0

            # finish time of the same trade in the previous zone
            prev_zone_finish = S[i-1, j] + D[i-1, j] if i > 0 else 0    

            S[i, j] = max(prev_trade_finish, prev_zone_finish)

    return S, D

def evaluate_schedule(z, c, S, D):

    num_zones, num_trades = S.shape

    # total duration of the schedule
    duration = np.max(S + D)

    # total idle time
    idle_time = 0.0
    for j in range(num_trades):
        for i in range(1, num_zones):
            previous_finish = S[i - 1, j] + D[i - 1, j]
            idle_time += S[i, j] - previous_finish

    # connectedness penalty
    room_boundaries_db = pd.read_csv("./data/Room_Boundaries.csv")
    room_boundaries_db = room_boundaries_db[room_boundaries_db["Level"].eq("L 1")]
    room_ids = set(map(int, room_boundaries_db["RoomId"].unique()))
    adjacency = {room_id: set() for room_id in room_ids}
    for _, boundary_df in room_boundaries_db.dropna(subset=["Boundary Element Id"]).groupby("Boundary Element Id"):
        adjacent_rooms = set(map(int, boundary_df["RoomId"].unique()))
        for room_id in adjacent_rooms:
            adjacency[room_id].update(adjacent_rooms - {room_id})

    zones = {}
    for room_id, zone_id in z.items():
        zones.setdefault(zone_id, set()).add(int(room_id))

    connectedness_penalty = 0
    for zone_rooms in zones.values():
        unseen = set(zone_rooms)
        num_components = 0
        while unseen:
            num_components += 1
            stack = [unseen.pop()]
            while stack:
                room_id = stack.pop()
                connected_rooms = adjacency[room_id] & unseen
                stack.extend(connected_rooms)
                unseen -= connected_rooms
        connectedness_penalty += num_components - 1

    # crew cost penalty
    crew_cost = sum(c.values())

    return [duration, idle_time, connectedness_penalty, crew_cost]
