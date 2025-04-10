import numpy as np
import pandas as pd
import shutil

from pathlib import Path


def create_folder(folder: Path = None, clear: bool = False):
    """ Create a new folder

    Parameters
    ----------
    folder: pathlib.Path
        The folder path.
    clear: bool
        Clear content if folder exists.
    """

    if folder:
        if folder.is_dir() and clear:
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def read_igrf13_data(file, extent=None):
    df = pd.read_csv(file,
                     sep='\\s+',
                     skipinitialspace=True,
                     skiprows=2)

    if extent:
        lon_min, lon_max, lat_min, lat_max = extent
        df = df[(df['Lat'] >= lat_min) & (df['Lat'] <= lat_max)]
        df = df[(df['Long'] >= lon_min) & (df['Long'] <= lon_max)]

    table = pd.pivot_table(df, values='I', index=['Lat'], columns=['Long'])
    table = np.rad2deg(np.arctan(np.tan(np.deg2rad(table)) * 0.5))

    return table
