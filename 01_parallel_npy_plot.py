import multiprocessing as mp
import numpy as np
import ray
import sys

from datetime import datetime
from pathlib import Path

from src.constants import TEC_MIN, TEC_MAX, MAP_EXTENT, PLOT_EXTENT, IGRF_EXTENT
from src.graphics import plot_tec_map
from src.util import create_folder, read_igrf13_data
from src.tecmap import Embrace, IGS, Maggia, Nagoya, TecMap


@ray.remote
def plot_br_region_tec_map(epoch, tec_map_obj: TecMap, output_dir):
    year = epoch.astype(datetime).strftime('%Y')

    network = (tec_map_obj.__class__.__name__).upper()
    if network == 'NAGOYA':
        tec_map, lon, lat = tec_map_obj.get_subset(MAP_EXTENT,
                                                   apply_lat_threshold=False)
    else:
        tec_map, lon, lat = tec_map_obj.get_subset(MAP_EXTENT)

    tec_map[tec_map < TEC_MIN] = TEC_MIN
    tec_map[tec_map > TEC_MAX] = TEC_MAX

    file_name = f"{str(epoch).replace(':', '.')}.png"
    plot_tec_map(tec_map, lon, lat, PLOT_EXTENT, igrf_table[year],
                 IGRF_EXTENT,f'{epoch} UT',
                 network, TEC_MIN, TEC_MAX, output_dir / file_name)


if __name__ == '__main__':

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

    years = ['2022', '2023', '2024']
    igrf_table = dict()
    for year in years:
        igrf_file = Path('.').resolve() / 'igrf' / f'{year}_low.txt'
        igrf_table[year] = read_igrf13_data(
            igrf_file, extent=IGRF_EXTENT)

    if 'maggia' in npy_file.lower():
        network_class = Maggia
    elif 'igs' in npy_file.lower():
        network_class = IGS
    elif 'nagoya' in npy_file.lower():
        network_class = Nagoya
    else:
        network_class = Embrace

    output_dir = Path('.').resolve() / 'output' / f'{npy_file[:-4]}_plot'
    create_folder(output_dir, clear=False)

    maps_array = np.load(data_dir / npy_file)
    print(maps_array.shape)
    print(maps_array[0])

    datetimes_list = np.load(data_dir / datetime_npy_file)
    print(datetimes_list.shape)
    print(datetimes_list[0])

    MAX_CPUS = mp.cpu_count()
    MAX_NUM_PENDING_TASKS = MAX_CPUS

    ray.init(num_cpus=MAX_CPUS)
    result_refs = []
    for i, tec_map in enumerate(maps_array):

        tec_map_obj = network_class()
        tec_map_obj.add_tec_map(tec_map)

        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            ray.get(ready_refs)

        result_refs.append(plot_br_region_tec_map.remote(datetimes_list[i],
                                                         tec_map_obj,
                                                         output_dir=output_dir))
    ray.get(result_refs)

