import multiprocessing as mp
import numpy as np
import pandas as pd
import ray
import sys

from datetime import datetime, date
from pathlib import Path

from src.constants import MAP_EXTENT
from src.utils import create_folder
from src.tecmap import Embrace, IGS, Maggia, Nagoya, TecMap


@ray.remote
def process_tec_map(epoch, tec_map_obj: TecMap, output_dir):
    year = epoch.astype(datetime).strftime('%Y')

    network = (tec_map_obj.__class__.__name__).upper()
    if network == 'NAGOYA':
        tec_map, lon, lat = tec_map_obj.get_subset(MAP_EXTENT,
                                                   apply_lat_threshold=False)
    else:
        tec_map, lon, lat = tec_map_obj.get_subset(MAP_EXTENT)

    print(epoch)
    subset_gopi_df = gopi_df.loc[epoch].copy()
    subset_gopi_df.rename(columns={'lat': 'latitude'}, inplace=True)
    subset_gopi_df = subset_gopi_df.join(rbmc_stations, on='station')

    tec_values = tec_map_obj.get_tec_value(subset_gopi_df['lon'],
                                           subset_gopi_df['lat'])
    subset_gopi_df['tec'] = tec_values

    subset_gopi_df = subset_gopi_df[['mean', 'std', 'tec', 'lat', 'lon',
                                     'station']].copy()
    subset_gopi_df['network'] = network
    subset_gopi_df.reset_index(drop=True, inplace=True)
    subset_gopi_df['datetime'] = epoch.astype(datetime).strftime(
        '%Y-%m-%d %H:%M:%S')

    subset_gopi_df.to_csv(output_dir / f"gopi-{network.lower()}-data_{epoch}.csv", index=False)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    print(f"Project dir: {project_dir}")

    if len(sys.argv) == 2:
        data_file = Path(sys.argv[1]).resolve()
    else:
        print("Data file not informed!")
        exit()

    if not data_file.exists():
        print("Data file not found!")

    data_dir = data_file.parent
    npy_file = data_file.name
    datetime_npy_file = npy_file[:-4] + '_datetimes.npy'

    if 'maggia' in npy_file.lower():
        network_class = Maggia
    elif 'igs' in npy_file.lower():
        network_class = IGS
    elif 'nagoya' in npy_file.lower():
        network_class = Nagoya
    else:
        network_class = Embrace

    output_dir = project_dir / 'output' / 'gopi+tecmaps'
    create_folder(output_dir, clear=False)

    maps_array = np.load(data_dir / npy_file)
    datetimes_list = np.load(data_dir / datetime_npy_file)

    MAX_CPUS = mp.cpu_count()
    MAX_NUM_PENDING_TASKS = MAX_CPUS
    START_DATE = date(2024, 12, 1)

    rbmc_stations = pd.read_json(project_dir / 'rbmc_stations.json')
    rbmc_stations = rbmc_stations[['name', 'lat', 'lon']]
    rbmc_stations.set_index('name', inplace=True)
    print(rbmc_stations)

    gopi_stations = []
    input_dir = project_dir / 'data' / 'gopi'
    for station_dir in sorted(input_dir.glob('Gopi_TEC_????_???_????')):
        gopi_stations.append(station_dir.name.split('_')[2])
    print(gopi_stations)

    input_dir = project_dir / 'output'
    gopi_df = pd.read_csv(input_dir / 'mean_std_gopi_dataset.csv',
                          parse_dates=['datetime'])
    gopi_df.set_index('datetime', inplace=True)
    print(gopi_df)
    print(gopi_df.index)

    ray.init(num_cpus=MAX_CPUS)
    result_refs = []
    for i, tec_map in enumerate(maps_array):
        # tec_map[tec_map > 999] = np.nan
        tec_map_obj = network_class()
        tec_map_obj.add_tec_map(tec_map)

        if datetimes_list[i].astype(datetime).date() < START_DATE:
            pass
        else:
            if len(result_refs) > MAX_NUM_PENDING_TASKS:
                ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
                ray.get(ready_refs)

            result_refs.append(process_tec_map.remote(datetimes_list[i], tec_map_obj, output_dir=output_dir))

    ray.get(result_refs)
