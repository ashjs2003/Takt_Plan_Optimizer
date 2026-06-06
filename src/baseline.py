import pandas as pd

from logger import plot_takt_plan, plot_zoning
from scheduler import evaluate_schedule, make_schedule


def zigzag_room_order(room_locations: pd.DataFrame) -> list[int]:
    rows = []
    for _, room in room_locations.sort_values("Room Location Y (ft)").iterrows():
        if not rows or abs(room["Room Location Y (ft)"] - rows[-1]["y"].mean()) > 12:
            rows.append({"y": pd.Series([room["Room Location Y (ft)"]]), "rooms": [room]})
        else:
            rows[-1]["y"] = pd.concat([rows[-1]["y"], pd.Series([room["Room Location Y (ft)"]])], ignore_index=True)
            rows[-1]["rooms"].append(room)

    ordered_rooms = []
    for row_index, row in enumerate(rows):
        row_rooms = pd.DataFrame(row["rooms"]).sort_values("Room Location X (ft)", ascending=row_index % 2 == 0)
        ordered_rooms.extend(row_rooms["RoomId"].astype(int).tolist())
    return ordered_rooms


def make_baseline_z(ordered_rooms: list[int], rooms_per_zone: int = 2) -> dict[int, int]:
    return {
        room_id: index // rooms_per_zone + 1
        for index, room_id in enumerate(ordered_rooms)
    }

def baseline_candidate():
    room_boundaries = pd.read_csv("./data/Room_Boundaries.csv")
    room_locations = room_boundaries[room_boundaries["Level"].eq("L 1")].groupby("RoomId").first().reset_index()
    trade_data = pd.read_csv("./data/Takt_Productivity_Rates.csv")

    z = make_baseline_z(zigzag_room_order(room_locations))
    c = {trade: 2 for trade in trade_data["trade"]}

    S, D = make_schedule(z, c)
    candidate = {
        "zones": z,
        "crews": c,
        "metrics": evaluate_schedule(z, c, S, D),
    }
    return candidate


if __name__ == "__main__":
    candidate = baseline_candidate()

    plot_zoning(candidate, "./outputs/baseline_zoning.png")
    plot_takt_plan(candidate, "./outputs/baseline_takt_plan.png")
    print(candidate["metrics"])
