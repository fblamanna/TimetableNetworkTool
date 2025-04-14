import pandas as pd
import numpy as np
import string
from datetime import datetime, timedelta

# -------------------------------
# PART 1: Generate Random Stations
# -------------------------------

def generate_station_code(existing):
    """
    Generate a unique 3-letter station code.
    """
    while True:
        code = ''.join(np.random.choice(list(string.ascii_uppercase), 3))
        if code not in existing:
            existing.add(code)
            return code

def generate_random_stations(num_stations=10, lat_range=(10, 50), lon_range=(10, 50)):
    """
    Generate a list of random stations along with random coordinates.
    Returns:
      - A list of station codes.
      - A DataFrame with columns.
      
    The latitude and longitude are uniformly distributed within the specified ranges.
    """
    station_set = set()
    stations = []
    lats = np.random.uniform(lat_range[0], lat_range[1], num_stations)
    lons = np.random.uniform(lon_range[0], lon_range[1], num_stations)
    
    for i in range(num_stations):
        code = generate_station_code(station_set)
        stations.append(code)

    stations_id = np.random.choice(len(stations), len(stations), replace=False)
    
    coords_df = pd.DataFrame({
        'Station ID': stations_id,
        'Station Code': stations,
        'Station Name': stations,
        'Longitude (degrees)': lons,
        'Latitude (degrees)': lats
    })
    return stations, coords_df

# -------------------------------
# PART 2: Generate Random Train Information
# -------------------------------

def generate_train_code(existing):
    """
    Generate a unique train code that starts with 'R' or 'E' followed by two random digits.
    """
    prefix = np.random.choice(['R', 'E'])
    while True:
        code = prefix + ''.join(np.random.choice(list(string.digits), 2))
        if code not in existing:
            existing.add(code)
            return code

def generate_random_trains(num_trains=5, departure_window=('05:00:00', '12:00:00')):
    """
    Generate random train information.
    For each train, a random code is generated along with a random first departure time and
    a last arrival time that is 1 to 2 hours after the departure.
    
    Returns:
      A list of lists: [train_code, first_departure (as HH:MM:SS), last_arrival (as HH:MM:SS)].
    """
    train_set = set()
    trains = []
    dep_start = datetime.strptime(departure_window[0], '%H:%M:%S')
    dep_end   = datetime.strptime(departure_window[1], '%H:%M:%S')
    dep_range_minutes = int((dep_end - dep_start).total_seconds() / 60)

    for _ in range(num_trains):
        train_code = generate_train_code(train_set)
        # Generate a random first departure within the departure window.
        first_dep_minutes = np.random.randint(0, dep_range_minutes)
        first_dep_time = dep_start + timedelta(minutes=first_dep_minutes)
        # Generate a random travel duration between 1 and 2 hours.
        travel_duration = np.random.randint(60, 120)  # minutes
        last_arrival_time = first_dep_time + timedelta(minutes=travel_duration)
        trains.append([train_code,
                       first_dep_time.strftime('%H:%M:%S'),
                       last_arrival_time.strftime('%H:%M:%S')])
    return trains

# -------------------------------
# PART 3: Generate Random Timetable
# -------------------------------

def generate_random_timetable(stations, trains_info, stop_probability=0.7, stop_duration_range=(1,3)):
    """
    Generate a timetable for each train.
    
    For each train:
      - A random subset (minimum 2) of the provided station list is chosen.
      - The order of the stations is randomized.
      - Arrival times for the intermediate stations are uniformly distributed between
        the train's first departure and last arrival.
      - For each intermediate station, the script randomly decides whether the train
        stops (adding a stop duration) or just passes through.
    
    Returns:
      A DataFrame with columns:
      ['Train number', 'Station', 'Arrival time', 'Departure time', 'Stop type'].
    """
    timetable = []
    for train, first_dep, last_arr in trains_info:
        dep_time = datetime.strptime(first_dep, '%H:%M:%S')
        arr_time = datetime.strptime(last_arr, '%H:%M:%S')
        total_seconds = int((arr_time - dep_time).total_seconds())
        
        # Randomly select a subset of stations (at least 2) for this train's route.
        num_train_stations = np.random.randint(2, len(stations) + 1)
        train_route = np.random.choice(stations, num_train_stations, replace=False).tolist()
        np.random.shuffle(train_route)  # Randomize the order of selected stations.
        
        n_stops = len(train_route)
        # Generate uniformly distributed times (in seconds) for intermediate stops.
        if n_stops > 2:
            random_offsets = sorted(np.random.uniform(0, total_seconds, n_stops - 2))
            station_offsets = [0] + random_offsets + [total_seconds]
        else:
            station_offsets = [0, total_seconds]
        
        # Add first station (begin)
        timetable.append([train, train_route[0], '', dep_time.strftime('%H:%M:%S'), 'begin'])
        
        # Add intermediate stations
        for i in range(1, n_stops - 1):
            arrival_time = dep_time + timedelta(seconds=int(station_offsets[i]))
            # Determine if the train stops or just passes through.
            if np.random.rand() < stop_probability:
                stop_type = 'stop'
                stop_minutes = np.random.randint(stop_duration_range[0], stop_duration_range[1] + 1)
            else:
                stop_type = 'pass'
                stop_minutes = 0
            departure_time = arrival_time + timedelta(minutes=stop_minutes)
            timetable.append([
                train,
                train_route[i],
                arrival_time.strftime('%H:%M:%S'),
                departure_time.strftime('%H:%M:%S') if stop_minutes else arrival_time.strftime('%H:%M:%S'),
                stop_type
            ])
        
        # Add last station (end)
        timetable.append([train, train_route[-1], arr_time.strftime('%H:%M:%S'), '', 'end'])
    
    timetable_df = pd.DataFrame(timetable, columns=['Train number', 'Station', 'Arrival time', 'Departure time', 'Stop type'])
    return timetable_df

# -------------------------------
# MAIN EXECUTION
# -------------------------------

if __name__ == "__main__":
    # Parameters (adjust as needed)
    num_stations = 10
    num_trains   = 5
    departure_window = ('05:00:00', '12:00:00')
    lat_range = (10, 50)
    lon_range = (10, 50)
    
    # Generate random stations with coordinates.
    stations, stations_df = generate_random_stations(num_stations=num_stations, lat_range=lat_range, lon_range=lon_range)
    stations_csv_path = "RandomStationCoordinates.csv"
    stations_df.to_csv(stations_csv_path, index=False, sep=';')
    print(f"Generated station coordinates saved to '{stations_csv_path}'")
    print(stations_df.head())
    
    # Generate random train information.
    trains_info = generate_random_trains(num_trains=num_trains, departure_window=departure_window)
    print("\nGenerated Train Information:")
    for train in trains_info:
        print(train)
    
    # Generate random timetable based on the generated stations and train info.
    timetable_df = generate_random_timetable(stations, trains_info, stop_probability=0.7, stop_duration_range=(1,3))
    timetable_csv_path = "RandomTimetable.csv"
    timetable_df.to_csv(timetable_csv_path, index=False, sep=';')
    print(f"\nGenerated timetable saved to '{timetable_csv_path}'")
    print("\nTimetable Preview:")
    print(timetable_df.head(20))
