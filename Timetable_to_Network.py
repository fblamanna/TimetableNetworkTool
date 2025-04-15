import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# =============================================================================
# FUNCTION: build_directed_network_with_time
# =============================================================================
def build_directed_network_with_time(timetable_df, space_type):
    """
    Build a directed weighted network from timetable data.
    
    For each directed edge (source → target), this function computes:
      - DSN: the count of occurrences (i.e. number of trains that run that directed link)
      - DTN: aggregates travel times (in seconds) and counts valid observations so that 
             the mean travel time (in minutes) can be computed and then its reciprocal taken.
    
    Parameters:
      timetable_df: DataFrame with columns "Train number", "Station", "Arrival time",
                    "Departure time", and "Stop type".
      space_type: one of:
         - "stations": allowed types {begin, pass, stop, end, service_stop}.
                       Build directed edges between consecutive stations in the full route.
         - "stops": allowed types {begin, stop, end}.
                    Build directed edges between consecutive stops.
         - "changes": allowed types {begin, stop, end}.
                      For each train, form a clique among the UNIQUE stops (duplicates removed using
                      normalized station names — trimmed and converted to uppercase). Then, for each
                      ordered pair (i, j) with i < j (following the train's order), add a directed edge
                      from station i to station j.
    
    Returns:
      vertices: set of station codes.
      edge_data: dict mapping (source, target) → {"dsn": int, "dt_sum": float, "dt_count": int}
    """
    if space_type == "stations":
        allowed = {"begin", "pass", "stop", "end", "service_stop"}
        consecutive_mode = True
        clique_mode = False
    elif space_type == "stops":
        allowed = {"begin", "stop", "end"}
        consecutive_mode = True
        clique_mode = False
    elif space_type == "changes":
        allowed = {"begin", "stop", "end"}
        consecutive_mode = False
        clique_mode = True
    else:
        raise ValueError("Unknown space type.")
    
    vertices = set()
    edge_data = {}  # key: (source, target)
    
    groups = timetable_df.groupby("Train number", sort=False)
    allowed_lower = {s.lower() for s in allowed}
    
    for train, group in groups:
        group_ordered = group.sort_index()
        filtered = group_ordered[group_ordered["Stop type"].str.lower().isin(allowed_lower)]
        rows = filtered.to_dict('records')
        
        if not clique_mode:
            # For consecutive modes ("stations" and "stops"), use all rows in order.
            station_list = [row["Station"].strip() for row in rows]
            for st in station_list:
                vertices.add(st)
            pairs = [(i, i + 1) for i in range(len(rows) - 1)]
            for i, j in pairs:
                src = rows[i]["Station"].strip()
                tgt = rows[j]["Station"].strip()
                edge = (src, tgt)
                if edge not in edge_data:
                    edge_data[edge] = {"dsn": 0, "dt_sum": 0.0, "dt_count": 0}
                edge_data[edge]["dsn"] += 1
                dep_time_str = rows[i].get("Departure time", "")
                arr_time_str = rows[j].get("Arrival time", "")
                if pd.notna(dep_time_str) and dep_time_str != "" and pd.notna(arr_time_str) and arr_time_str != "":
                    try:
                        dep_time = datetime.strptime(dep_time_str, "%H:%M:%S")
                        arr_time = datetime.strptime(arr_time_str, "%H:%M:%S")
                        travel_time = (arr_time - dep_time).total_seconds()
                        if travel_time >= 0:
                            edge_data[edge]["dt_sum"] += travel_time
                            edge_data[edge]["dt_count"] += 1
                    except Exception:
                        pass
        else:
            # For "changes" mode: deduplicate stops in the train using normalized station names.
            unique_rows = []
            seen = set()
            for row in rows:
                norm = row["Station"].strip().upper()
                if norm not in seen:
                    unique_rows.append(row)
                    seen.add(norm)
            for row in unique_rows:
                vertices.add(row["Station"].strip())
            n = len(unique_rows)
            for i in range(n):
                for j in range(i + 1, n):
                    src = unique_rows[i]["Station"].strip()
                    tgt = unique_rows[j]["Station"].strip()
                    # Compute travel time using the natural order (if available)
                    dep_time_str = unique_rows[i].get("Departure time", "")
                    arr_time_str = unique_rows[j].get("Arrival time", "")
                    travel_time = None
                    if pd.notna(dep_time_str) and dep_time_str != "" and pd.notna(arr_time_str) and arr_time_str != "":
                        try:
                            dep_time = datetime.strptime(dep_time_str, "%H:%M:%S")
                            arr_time = datetime.strptime(arr_time_str, "%H:%M:%S")
                            diff = (arr_time - dep_time).total_seconds()
                            if diff >= 0:
                                travel_time = diff
                        except Exception:
                            travel_time = None
                    # Only add the edge in the natural order: src -> tgt.
                    edge = (src, tgt)
                    if edge not in edge_data:
                        edge_data[edge] = {"dsn": 0, "dt_sum": 0.0, "dt_count": 0}
                    edge_data[edge]["dsn"] += 1
                    if travel_time is not None:
                        edge_data[edge]["dt_sum"] += travel_time
                        edge_data[edge]["dt_count"] += 1
                    # Note: We do not add a reverse edge here, so direction is preserved.
    
    return vertices, edge_data

# =============================================================================
# FUNCTION: write_pajek_arcs_with_mode
# =============================================================================
def write_pajek_arcs_with_mode(vertices, edge_data, filename, mode):
    """
    Write the directed network to a Pajek file.
    
    The file header uses "*Vertices" and "*Arcs <total_edges>".
    
    Each arc is written as: vertexID_source vertexID_target weight
      - If mode is "dsn": weight = DSN (the number of trains for that directed link)
      - If mode is "dtn": weight = reciprocal of the mean travel time (in minutes)
                     (if the mean travel time is T > 0 then weight = 1/T, else 0).
    
    Parameters:
      vertices: set of station codes.
      edge_data: dict mapping (source, target) → {"dsn", "dt_sum", "dt_count"}.
      filename: output file name.
      mode: either "dsn" or "dtn".
    """
    vertex_list = sorted(list(vertices))
    vertex_map = {station: i + 1 for i, station in enumerate(vertex_list)}
    total_edges = len(edge_data)
    
    with open(filename, "w") as f:
        f.write(f"*Vertices {len(vertex_list)}\n")
        for station, vid in vertex_map.items():
            f.write(f'{vid} "{station}"\n')
        f.write(f"*Arcs {total_edges}\n")
        for (src, tgt), data in edge_data.items():
            if mode == "dsn":
                weight = data["dsn"]
                f.write(f"{vertex_map[src]} {vertex_map[tgt]} {weight}\n")
            elif mode == "dtn":
                if data["dt_count"] > 0:
                    mean_dt_minutes = (data["dt_sum"] / data["dt_count"]) / 60.0
                else:
                    mean_dt_minutes = 0.0
                if mean_dt_minutes > 0:
                    weight = 1.0 / mean_dt_minutes
                    #weight = mean_dt_minutes
                else:
                    weight = 0.0
                f.write(f"{vertex_map[src]} {vertex_map[tgt]} {weight:.2f}\n")
            else:
                raise ValueError("Invalid mode. Use 'dsn' or 'dtn'.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    # Read the input timetable CSV file (semicolon-separated).
    input_file = "input.csv"
    timetable_df = pd.read_csv(input_file, sep=';')
    
    # Define spaces: tuple = (space_type, DSN output filename, DTN output filename)
    spaces = {
        "Space of Stations": ("stations", "DSN_SpaceStations.net", "DTN_SpaceStations.net"),
        "Space of Stops": ("stops", "DSN_SpaceStops.net", "DTN_SpaceStops.net"),
        "Space of Changes": ("changes", "DSN_SpaceChanges.net", "DTN_SpaceChanges.net")
    }
    
    for space_name, (space_type, dsn_filename, dtn_filename) in spaces.items():
        vertices, edge_data = build_directed_network_with_time(timetable_df, space_type)
        write_pajek_arcs_with_mode(vertices, edge_data, dsn_filename, mode="dsn")
        write_pajek_arcs_with_mode(vertices, edge_data, dtn_filename, mode="dtn")
        print(f"{space_name} DSN network saved to: {dsn_filename}")
        print(f"{space_name} DTN network saved to: {dtn_filename}")

if __name__ == "__main__":
    main()
