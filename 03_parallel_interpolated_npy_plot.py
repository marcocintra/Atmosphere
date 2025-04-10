
import multiprocessing as mp
import numpy as np
import ray
import sys

from pathlib import Path

from src.constants import (TEC_MIN, TEC_MAX, MAP_EXTENT, PLOT_EXTENT,
                           IGRF_EXTENT, INTERP_STEP)
from src.graphics import plot_tec_map, plot_tec_map_raster
from src.util import create_folder, read_igrf13_data
from src.tecmap import Embrace, IGS, Maggia, Nagoya, TecMap


@ray.remote
def plot_interpolated_tec_map(epoch,
                              tec_map_obj: TecMap,
                              output_dir,
                              output_raster_dir):
    print(epoch)
    year = str(np.datetime64(epoch.replace('.', ':'), 'Y'))

    network = (tec_map_obj.__class__.__name__).upper()
    tec_map, lon, lat = tec_map_obj.get_subset(MAP_EXTENT)

    tec_map[tec_map < TEC_MIN] = TEC_MIN
    tec_map[tec_map > TEC_MAX] = TEC_MAX


    file_name = f"{epoch}.png"
    plot_tec_map(tec_map, lon, lat, PLOT_EXTENT, igrf_table[year],
                 IGRF_EXTENT,f"{epoch.replace('.', ':')} UT",
                 network, TEC_MIN, TEC_MAX, output_dir / file_name)

    file_name = f"{epoch}_raster.png"
    plot_tec_map_raster(tec_map, TEC_MIN, TEC_MAX,
                        output_raster_dir / file_name)



if __name__ == '__main__':

    if len(sys.argv) == 2:
        full_data_dir = Path(sys.argv[1]).resolve()
    else:
        print("Data dir not informed!")
        exit()
    print(full_data_dir)
    if not full_data_dir.exists():
        print("Data dir not found!")

    data_dir = full_data_dir.parent
    npy_dir = full_data_dir.name

    years = ['2022', '2023', '2024']
    igrf_table = dict()
    for year in years:
        igrf_file = Path('.').resolve() / 'igrf' / f'{year}_low.txt'
        igrf_table[year] = read_igrf13_data(
            igrf_file, extent=IGRF_EXTENT)

    if 'maggia' in npy_dir.lower():
        network_class = Maggia
    elif 'igs' in npy_dir.lower():
        network_class = IGS
    elif 'nagoya' in npy_dir.lower():
        network_class = Nagoya
    else:
        network_class = Embrace

    output_dir = Path('.').resolve() / 'output' / f'{npy_dir}_plot'
    create_folder(output_dir, clear=False)

    output_raster_dir = Path('.').resolve() / 'output' / f'{npy_dir}_raster'
    create_folder(output_raster_dir, clear=False)


    MAX_CPUS = mp.cpu_count()
    MAX_NUM_PENDING_TASKS = MAX_CPUS

    ray.init(num_cpus=MAX_CPUS)
    result_refs = []
    for file in (data_dir / npy_dir).glob('*.npy'):
        epoch = file.name[:-4]
        tec_map = np.load(file)
        tec_map_obj = network_class(extent=MAP_EXTENT,
                                    lat_step=INTERP_STEP,
                                    lon_step=INTERP_STEP)
        tec_map_obj.add_tec_map(tec_map)

        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            ray.get(ready_refs)

        result_refs.append(
            plot_interpolated_tec_map.remote(
                epoch,
                tec_map_obj,
                output_dir=output_dir,
                output_raster_dir=output_raster_dir
            )
        )
    ray.get(result_refs)

