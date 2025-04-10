import gc
import numpy as np
import pandas as pd
import sys

from pathlib import Path

from src.constants import MAP_EXTENT, INTERP_STEP
from src.interpolation import InverseDistanceWeightingInterpolation
from src.util import create_folder
from src.tecmap import Embrace, IGS, Maggia, Nagoya, TecMap


def interpolate(epoch, tec_map_obj: TecMap, radius, output_dir):
    print(epoch)
    apply_lat_threshold = True
    apply_lon_threshold = True

    if type(tec_map_obj) == IGS:
        threshold = 20
    elif type(tec_map_obj) == Embrace:
        threshold = 10
    elif type(tec_map_obj) == Nagoya:
        threshold = 5
        apply_lat_threshold = False
    else:
        threshold = 5

    sub_tec_map, sub_lon, sub_lat = tec_map_obj.get_subset(MAP_EXTENT,
                                                           threshold,
                                                           apply_lat_threshold,
                                                           apply_lon_threshold)

    print(sub_tec_map.shape)

    tec_map_df = pd.DataFrame(sub_tec_map,
                              index=sub_lat,
                              columns=sub_lon)
    tec_map_df.reset_index(inplace=True)
    tec_map_df.rename(columns={'index': 'latitude'}, inplace=True)
    tec_map_df = tec_map_df.melt(id_vars='latitude', var_name='longitude',
                                 value_name='tec')
    tec_map_df = tec_map_df.dropna(subset=['tec'])

    interp = InverseDistanceWeightingInterpolation(extent=MAP_EXTENT,
                                                   step=INTERP_STEP)
    interp.interpolate(tec_map_df['longitude'].astype('float'),
                       tec_map_df['latitude'].astype('float'),
                       tec_map_df['tec'].astype('float'), r=radius, reg=1e-10)

    del tec_map_df
    gc.collect()

    tec_map = interp.interpolated_map

    # tec_map[tec_map <= 0] = 0

    if type(tec_map_obj) != IGS:
        tec_map = np.flip(tec_map, axis=0)

    file_name = f"{str(epoch).replace(':', '.')}.npy"
    np.save(output_dir / file_name, tec_map)

    return output_dir / file_name



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

    if 'maggia' in npy_file.lower():
        network_class = Maggia
        radius = 250
    elif 'igs' in npy_file.lower():
        network_class = IGS
        radius = 1500
    elif 'nagoya' in npy_file.lower():
        network_class = Nagoya
        radius = 250
    else:
        network_class = Embrace
        radius = 500

    output_dir = Path('.').resolve() / 'output' / f'{npy_file[:-4]}_interp'
    create_folder(output_dir, clear=False)

    maps_array = np.load(data_dir / npy_file)
    print(maps_array.shape)
    print(maps_array[0])

    datetimes_list = np.load(data_dir / datetime_npy_file)
    print(datetimes_list.shape)
    print(datetimes_list[0])

    for i, tec_map in enumerate(maps_array):
        tec_map_obj = network_class()
        tec_map_obj.add_tec_map(tec_map)

        interpolate(datetimes_list[i], tec_map_obj, radius=radius,
                    output_dir=output_dir)


