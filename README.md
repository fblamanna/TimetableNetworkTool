# Timetable Network Tool
The documentation refers to the pre-print paper published on [ArXiv](https://arxiv.org/abs/2504.09214) - *Unveiling connectivity patterns of railway timetables through complex network theory and Infomap clustering*.
The documentation is organized as follows:

- **Random Train Timetable Generator**: the `TimetableGenerator.py` script returns a .csv file with a Synthetic Timetable with train paths, stations, departure and arrival times. The resulting file is a potential input of the following script.

- **Timetables to Networks**: the `Timetables_to_Networks.py` script get as input a Timetable structure file (from real data or generated by the above mentioned script) and creates several networks based on [Spaces](https://arxiv.org/abs/physics/0510151) and Weights in [.pajek](http://mrvar.fdv.uni-lj.si/pajek/) files that may be upload and analysed with [Infomap Online](https://www.mapequation.org/infomap/) with the following configuration input:

`./ --no-self-links --directed --clu -N 10`

This creates output files over 10 different runs.

## Random Train Timetable Generator

### Overview

This is a Python script that randomly generates a set of station codes with corresponding coordinates and a randomized train timetable. The script creates two output CSV files:

- **`RandomStationCoordinates.csv`**: contains the list of generated station codes along with uniformly distributed random coordinates (latitude and longitude).
- **`RandomTimetable.csv`**: contains a randomized timetable where each train connects a randomly chosen subset of the stations. For each train, arrival and departure times at intermediate stations are uniformly distributed between the first departure time and the last arrival time. At each intermediate station, the train either stops (with a configurable delay) or passes through based on a defined probability.

### Features

- **Random Station Generation:**  
  Generates unique 3-letter station codes and assigns random coordinates (within defined latitude and longitude ranges).

- **Random Train Information:**  
  Generates random train codes (starting with "R" for regional or "E" for express), and assigns random first departure and last arrival times (with a travel duration between 1 and 10 hours) within a specified departure window.

- **Random Timetable Generation:**  
  For each train:
  - A random subset (minimum 2) of generated stations is chosen as the route.
  - The order of the selected stations is randomized.
  - Arrival and departure times for intermediate stations are uniformly distributed between the train's departure and arrival times.
  - For each intermediate station, a stop (with a delay) or a pass is randomly determined by the **stop probability** parameter.

### Parameters
User can adjust the following parameters to generate the random timetable data:

1. `num_stations`
2. `num_trains`
3. `departure_window`: e.g. ('05:00:00', '12:00:00') first and last departure time
4. `lat_range`: e.g. (10, 50) range of station coordinates (useful to plotting functions)
5. `lon_range`: e.g. (10, 50) range of station coordinates (useful to plotting functions)

### Requirements

- Python 3.x
- Dependencies: `pandas`, `numpy`

### Execution
Run the script with:
`python TimetableGenerator.py`

## From Timetables to Networks
This Python script generates directed network graphs (in Pajek format) from an input railway timetable CSV file. The script creates three distinct network "spaces"—each representing a different level of connection between stations—and produces two output files per space:

- **DSN (Directed Space Number):** The weight on each directed edge corresponds to the number of trains running that specific link.
- **DTN (Directed Travel Network):** The weight on each directed edge is the reciprocal of the mean travel time (in minutes) for that link.

### Network Spaces

The script supports three network spaces:

1. **Space of Stations:**  
   - **Definition:** Two stations are connected if they appear consecutively in a train's full route.  
   - **Allowed Stop Types:** `begin`, `pass`, `stop`, `end`, `service_stop`.

2. **Space of Stops:**  
   - **Definition:** Two stations are connected if they are consecutive when considering only stops.  
   - **Allowed Stop Types:** `begin`, `stop`, `end`.

3. **Space of Changes:**  
   - **Definition:** For each train, the stops (after deduplication using normalized station names) are considered in their natural order. A directed edge is created from station A to B only if the train runs from A to B. The reverse edge is added only if another train runs in that direction.  
   - **Allowed Stop Types:** `begin`, `stop`, `end`.

### How It Works

1. **Input File:**  
   The script reads a semicolon-separated CSV file (`input.csv`) with at least the following columns:  
   - `Train number`
   - `Station`
   - `Arrival time` (format: HH:MM:SS)
   - `Departure time` (format: HH:MM:SS)
   - `Stop type`

2. **Weights Calculation:**  
   - **DSN Weight:** Incremented for each occurrence of a directed edge.  
   - **DTN Weight:** Calculated as the reciprocal of the mean travel time (in minutes) for that edge. If a train’s travel time cannot be determined, its contribution is skipped.

4. **Output Files:**  
   For each of the three spaces, the script generates two files in Pajek `.net` format:
   - A file with DSN weights (e.g., `DSN_SpaceChanges.net`).
   - A file with DTN weights (e.g., `DTN_SpaceChanges.net`).

   The Pajek files contain a header with the vertex list and an arc section where each line specifies:
   ```
   vertexID_source vertexID_target weight
   ```

### How to Run

1. Place your input timetable CSV as `input.csv` in the same directory as the script.
2. Run the script using:
   ```bash
   python Timetable_to_Networks.py
   ```
3. Six output files (two per space) will be generated in the same directory.

### Dependencies

- Python 3.x
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

### Execution
Run the script with:
`python Timetables_to_Networks.py`

## Questions / Issues
Contacts and Info: [www.fabiolamanna.it](https://www.fabiolamanna.it)

##
Copyright (c) 2025 Fabio Lamanna. Code under License GPLv3.
