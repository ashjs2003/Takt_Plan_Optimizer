from pathlib import Path
import pandas as pd

def get_trade_bim_elements(bim_df: pd.DataFrame, trade: str) -> pd.DataFrame:

    if trade in {"Interior Walls", "Interior Finishes"}:
        interior_walls = bim_df["Category"].eq("Walls") & bim_df["Type"].str.contains("interior", case=False, na=False)
        return bim_df[interior_walls]
    if trade == "Doors":
        return bim_df[bim_df["Category"].eq("Doors")]
    if trade == "Ceiling":
        return bim_df[bim_df["Category"].eq("Ceilings") | (
            bim_df["Category"].eq("Parts") & bim_df["Original Category"].eq("Ceilings")
        )]
    if trade == "MEP":
        return bim_df[bim_df["Category"].isin({
            "Air Terminals",
            "Duct Fittings",
            "Ducts",
            "Electrical Fixtures",
            "Flex Ducts",
            "Mechanical Equipment",
            "Plumbing Fixtures",
            "Runs",
        })]
    return 

def numeric_quantity(value) -> float:
    if pd.isna(value):
        return 0.0
    quantity = str(value).split()[0]
    if quantity.lower() == "nan":
        return 0.0
    return float(quantity)

# raycasting logic
def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def calculate_quantity(bim_elements: pd.DataFrame, room_boundary: pd.Series, unit: str) -> float:
    polygon = room_boundary["Polygon"]
    boundary_element_ids = room_boundary["Boundary Element Ids"]

    elements = bim_elements.copy()
    elements = elements[elements["Level"].eq(room_boundary["Level"])]

    x = elements["Bounding Box Center X (ft)"].fillna(elements["Position X (ft)"])
    y = elements["Bounding Box Center Y (ft)"].fillna(elements["Position Y (ft)"])
    in_polygon = [
        point_in_polygon(px, py, polygon) if pd.notna(px) and pd.notna(py) else False
        for px, py in zip(x, y)
    ]

    in_polygon = pd.Series(in_polygon, index=elements.index)
    in_boundary = elements["ElementId"].isin(boundary_element_ids)
    room_elements = elements[in_polygon | in_boundary]

    quantity_unit = unit.split("/")[0]
    if quantity_unit == "EA":
        return float(len(room_elements))
    if quantity_unit == "SF":
        return sum(room_elements["Area"].apply(numeric_quantity), 0.0)
    return 0.00

if __name__ == "__main__":
    bim_df = pd.read_csv("./data/central_bim_model.csv")
    room_boundaries_df = pd.read_csv("./data/Room_Boundaries.csv")
    trade_df = pd.read_csv( "./data/Takt_Productivity_Rates.csv")
    bim_df = bim_df[bim_df["Level"].eq("L 1")]
    room_boundaries_df = room_boundaries_df[room_boundaries_df["Level"].eq("L 1")]

    # create the room polygons
    room_boundaries = []
    for _, room_df in room_boundaries_df.groupby("RoomId"):
        room_df = room_df.sort_values(["Boundary Loop", "Segment Index"])
        polygon = list(zip(room_df["Start X (ft)"], room_df["Start Y (ft)"]))
        room_boundaries.append(pd.Series({
            "RoomId": room_df.iloc[0]["RoomId"],
            "Level": room_df.iloc[0]["Level"],
            "Polygon": polygon,
            "Boundary Element Ids": set(pd.to_numeric(room_df["Boundary Element Id"], errors="coerce").dropna().astype(int)),
        }))

    # make a quantities db with columns as "room_id", "trade", "quantity", "unit"
    quantities_db = []
    for row in room_boundaries:
        for _, trade_row in trade_df.iterrows():
            trade = trade_row["trade"]
            bim_elements = get_trade_bim_elements(bim_df, trade)
            bim_quantity = calculate_quantity(bim_elements, row, trade_row["unit"])
            quantities_db.append({
                "room_id": row["RoomId"],
                "level": row["Level"],
                "trade": trade,
                "quantity": bim_quantity,
                "unit": trade_row["unit"],
            })

    quantities_db_df = pd.DataFrame(quantities_db)
    quantities_db_df.to_csv("./data/quantities_db.csv", index=False)
