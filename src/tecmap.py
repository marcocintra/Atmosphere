import numpy as np

from typing import Iterable, Any


class TecMap:

    def __init__(self, extent: Iterable[float], lat_step: float = 1,
                 lon_step: float = 1):

        self.lon_min, self.lon_max, self.lat_min, self.lat_max = extent
        self._lat_step = lat_step
        self._lon_step = lon_step

        self.lat = np.flip(np.arange(self.lat_min,
                                     self.lat_max + self._lat_step,
                                     self._lat_step, dtype='float'))
        self.lon = np.arange(self.lon_min,
                             self.lon_max + self._lon_step,
                             self._lon_step, dtype='float')

        self._tec_map = None

    def add_tec_map(self, tec_map: np.ndarray):
        self._tec_map = tec_map

    @property
    def tec_map(self):
        return self._tec_map

    @property
    def lat_step(self):
        return self._lat_step

    @property
    def lon_step(self):
        return self._lon_step

    def get_subset(self, subset_extent, threshold: float = 0,
                   apply_lat_threshold=True, apply_lon_threshold=True):
        print(self._tec_map.shape)

        lon_min, lon_max, lat_min, lat_max = subset_extent

        lat_step = self.lat_step
        lon_step = self.lon_step

        lat_threshold = np.ceil(threshold / lat_step) * lat_step
        lon_threshold = np.ceil(threshold / lon_step) * lon_step

        lon_min = np.floor(lon_min / lon_step) * lon_step - lon_threshold
        lon_max = np.ceil(lon_max / lon_step) * lon_step + lon_threshold
        lat_min = np.floor(lat_min / lat_step) * lat_step - lat_threshold
        lat_max = np.ceil(lat_max / lat_step) * lat_step + lat_threshold

        if lon_min < self.lon_min:
            lon_min = self.lon_min
        if lon_max > self.lon_max:
            lon_max = self.lon_max
        if lat_min < self.lat_min:
            lat_min = self.lat_min
        if lat_max > self.lat_max:
            lat_max = self.lat_max

        if not apply_lat_threshold:
            lat_min = self.lat_min
            lat_max = self.lat_max
        if not apply_lon_threshold:
            lon_min = self.lon_min
            lon_max = self.lon_max

        sub_lat = np.flip(np.arange(lat_min,
                                    lat_max + lat_step,
                                    lat_step,
                                    dtype='float'))
        sub_lon = np.arange(lon_min,
                            lon_max + lon_step,
                            lon_step,
                            dtype='float')

        rows = np.arange(self._latitude_to_index(lat_max),
                         self._latitude_to_index(lat_min)+1)
        cols = np.arange(self._longitude_to_index(lon_min),
                         self._longitude_to_index(lon_max)+1)

        return self._tec_map[np.ix_(rows, cols)], sub_lon, sub_lat

    def _longitude_to_index(self, longitude):
        try:
            return np.where(
                abs(np.array(self.lon) - longitude) < 0.001)[0][0]
        except IndexError as ex:
            return np.nan

    def _latitude_to_index(self, latitude):
        try:
            return np.where(
                abs(np.array(self.lat) - latitude) < 0.001)[0][0]
        except IndexError as ex:
            return np.nan


class Embrace(TecMap):

    def __init__(self,
                 extent: Iterable[float] = (-90, -30, -60, 20),
                 lat_step: float = 2,
                 lon_step: float = 2.5):
        super().__init__(extent, lat_step, lon_step)

    def add_tec_map(self, tec_map: np.ndarray):
        tec_map = np.flip(tec_map, axis=0)
        super().add_tec_map(tec_map)


class IGS(TecMap):

    def __init__(self,
                 extent: Iterable[float] = (-110, 0, -80, 40),
                 lat_step: float = 2.5,
                 lon_step: float = 5):
        super().__init__(extent, lat_step, lon_step)


class Maggia(TecMap):

    def __init__(self,
                 extent: Iterable[float] = (-110, 0, -80, 40),
                 lat_step: float = 0.5,
                 lon_step: float = 0.5):
        super().__init__(extent, lat_step, lon_step)

    def add_tec_map(self, tec_map: np.ndarray):
        tec_map = np.flip(tec_map, axis=0)
        super().add_tec_map(tec_map)

class Nagoya(TecMap):

    def __init__(self,
                 extent: Iterable[float] = (-110, 0, -80.4, 40.1),
                 lat_step: float = 0.5,
                 lon_step: float = 0.5):
        super().__init__(extent, lat_step, lon_step)

    def add_tec_map(self, tec_map: np.ndarray):
        tec_map = np.flip(tec_map, axis=0)
        super().add_tec_map(tec_map)