import numpy as np

from numba import njit
from typing import Iterable


class Interpolation:

    def __init__(self, extent: Iterable[float], step: float = 1):
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = extent
        self.step = step

        longitudes = np.arange(self.lon_min, self.lon_max+self.step, self.step)
        latitudes = np.arange(self.lat_min, self.lat_max+self.step, self.step)

        grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
        self._shape = grid_lon.shape

        self.X = np.c_[grid_lon.ravel(), grid_lat.ravel()]
        self.Y = None
        self.X_train = None
        self.Y_train = None

        self._interpolated_map = None

    def interpolate(self,
                    known_longitudes: Iterable[float],
                    known_latitudes: Iterable[float],
                    known_values: Iterable[float],
                    **kwargs):
        self.X_train = np.stack((known_longitudes, known_latitudes), axis=-1)
        self.Y_train = known_values

    @property
    def shape(self):
        return self._shape

    @property
    def interpolated_map(self) -> np.ndarray:
        return np.flip(self.Y.reshape(self.shape), axis=0)

    @staticmethod
    def haversine_distance(xa, xb, r=6335.439):
        lon_1 = xa[:, 0]*np.pi/180
        lat_1 = xa[:, 1]*np.pi/180
        lon_2 = xb[:, 0]*np.pi/180
        lat_2 = xb[:, 1]*np.pi/180

        return Interpolation._calc_haversine_distance(lon_1, lat_1,
                                                      lon_2, lat_2, r)

    @staticmethod
    @njit
    def _calc_haversine_distance(lon_1, lat_1, lon_2, lat_2, r):
        d_lon = np.abs(lon_1 - lon_2)

        a = np.power(np.cos(lat_2)*np.sin(d_lon), 2)
        b = np.power(np.cos(lat_1)*np.sin(lat_2) -
                     np.sin(lat_1)*np.cos(lat_2)*np.cos(d_lon), 2)
        c = (np.sin(lat_1)*np.sin(lat_2) +
             np.cos(lat_1)*np.cos(lat_2)*np.cos(d_lon))

        return np.abs(r*np.arctan(np.sqrt(a + b) / c))

    @staticmethod
    def cdist(xa, xb):
        xb_lon, xa_lon = np.meshgrid(xb[:, 0], xa[:, 0])
        xb_lat, xa_lat = np.meshgrid(xb[:, 1], xa[:, 1])

        xa = np.c_[xa_lon.ravel(), xa_lat.ravel()]
        xb = np.c_[xb_lon.ravel(), xb_lat.ravel()]

        dists = Interpolation.haversine_distance(xa, xb)
        return dists.reshape(xb_lon.shape)

    @staticmethod
    def pdist(x):
        def nump2(n, k):
            a = np.ones((k, n - k + 1), dtype=int)
            a[0] = np.arange(n - k + 1)
            for j in range(1, k):
                reps = (n - k + j) - a[j - 1]
                a = np.repeat(a, reps, axis=1)
                ind = np.add.accumulate(reps)
                a[j, ind[:-1]] = 1 - reps[1:]
                a[j, 0] = j
                a[j] = np.add.accumulate(a[j])
            return a

        def get_arrays(x):
            index = nump2(len(x), 2)
            a_index = index[0, :]
            b_index = index[1, :]

            return x[a_index], x[b_index]

        xa, xb = get_arrays(x)
        return Interpolation.haversine_distance(xa, xb)


class InverseDistanceWeightingInterpolation(Interpolation):
    def interpolate(self,
                    known_longitudes: Iterable[float],
                    known_latitudes: Iterable[float],
                    known_values: Iterable[float],
                    r=990,
                    reg=1e-10,
                    p=2):
        super().interpolate(known_longitudes, known_latitudes, known_values)

        distances = Interpolation.cdist(self.X_train, self.X)
        weights = (np.maximum(r - distances, 0) / (r * distances + reg)) ** p
        weights /= weights.sum(axis=0) + reg

        self.Y = np.dot(weights.T, self.Y_train)