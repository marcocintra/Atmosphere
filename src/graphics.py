import cartopy.crs as ccrs
import cartopy.feature as cfeatures
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from matplotlib.figure import Figure

line_style = dict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

def create_world_map(ax, extent=(-180, 180, -90, 90), figsize=(8, 8),
                     color='gray', fig=None,
                     show_xlabel=True, show_ylabel=True, fontsize=8,
                     linewidth=0.5):

    ax.set_extent(extent)

    land = cfeatures.LAND.with_scale('50m')
    ax.add_feature(land, facecolor='none', edgecolor=color, linewidth=linewidth)

    countries = cfeatures.BORDERS.with_scale('50m')
    ax.add_feature(countries, edgecolor=color, linestyle='-', linewidth=linewidth)

    states = cfeatures.STATES.with_scale('50m')
    ax.add_feature(states, edgecolor=color, linestyle='-', linewidth=linewidth)

    gl = ax.gridlines(draw_labels=True, linewidth=1, color=color,
                      linestyle=line_style['dotted'])
    gl.top_labels = False
    gl.right_labels = False

    if show_xlabel:
        gl.xformatter = LongitudeFormatter(direction_label=False)
        gl.xlocator = mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10])
        gl.xlabel_style = {'size': fontsize, 'weight': 'normal'}
    else:
        gl.bottom_labels = False

    if show_ylabel:
        gl.yformatter = LatitudeFormatter(direction_label=False)
        gl.ylocator = mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10])
        gl.ylabel_style = {'size': fontsize, 'weight': 'normal'}
    else:
        gl.left_labels = False

    gl.xlines = False
    gl.ylines = False


def plot_igrf(ax, table, extent=(-180, 180, -90, 90),
              levels=(-30, -20, -10, 0, 10, 20, 30),
              line_widths=(1, 1.5, 1, 2, 1, 1.5, 1), step=1, color='dimgray',
              fontsize=8):
    def fmt_igrf_latitude(x):
        s = f'{x:.1f}'
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s} ^{{\circ}}" if plt.rcParams["text.usetex"] else \
            rf"${s}^{{\mathbf{{\circ}}}}$"

    lon_min, lon_max, lat_min, lat_max = extent

    lat = np.arange(lat_min, lat_max + step, step, dtype='float')
    lon = np.arange(lon_min, lon_max + step, step, dtype='float')

    grid_lon, grid_lat = np.meshgrid(lon, lat)

    contour = ax.contour(grid_lon,
                         grid_lat,
                         table.values,
                         levels=levels,
                         # colors=['blue', 'blue', 'blue', 'green', 'red',
                         #         'red', 'red'],
                         colors=color,
                         alpha=1,
                         linewidths=line_widths,
                         zorder=2)
    ax.clabel(contour, inline=True, inline_spacing=4, fontsize=fontsize,
              fmt=fmt_igrf_latitude)

def plot_tec_map(tec_map, lon, lat, plot_extent, igrf_table, igrf_extent,
                 title='TEC Map', network='MAGGIA', tec_min=0, tec_max=160,
                 output_file='tec_map.png'):

    grid_lon, grid_lat = np.meshgrid(lon, lat)

    fig = Figure(figsize=(12.3, 10.8))

    ax = fig.subplots(1, subplot_kw=dict(
        projection=ccrs.PlateCarree()))
    create_world_map(ax,
                     plot_extent,
                     color='black', fontsize=18, linewidth=1)

    plot_igrf(ax,
              igrf_table,
              extent=igrf_extent,
              color='black', fontsize=18)

    cmap = mpl.colormaps.get_cmap("jet").copy()
    cmap.set_under('w', alpha=0)

    map = ax.pcolormesh(grid_lon, grid_lat, tec_map,
                        vmin=float(tec_min),
                        vmax=float(tec_max),
                        cmap=cmap,
                        alpha=1,
                        zorder=-2,
                        edgecolors='none',
                        antialiased=True,
                        shading='gouraud')

    ax.set_xticks([-80, -70, -60, -50, -40, -30], [],
                  crs=ccrs.PlateCarree())
    ax.axes.xaxis.set_ticklabels([])
    ax.set_yticks([-40, -30, -20, -10, 0, 10], [],
                  crs=ccrs.PlateCarree())
    ax.axes.yaxis.set_ticklabels([])

    ax.set_aspect(1)

    ax.text(0.5, -0.054, 'Geographic Longitude', transform=ax.transAxes,
            ha='center', va='top', fontsize=18)
    ax.text(-0.12, 0.5, 'Geographic Latitude', transform=ax.transAxes,
            rotation='vertical', va='center', fontsize=18)
    fig.text(0.97,
             0.022,
             f"[{np.nanmin(tec_map):.2f}, {np.nanmax(tec_map):.2f}] {network}",
             ha='right',
             va='bottom',
             fontsize=18)
    fig.suptitle(title, fontsize=24)

    cb = fig.colorbar(map, location='right',
                      ticks=np.arange(tec_min, tec_max+1, 20), shrink=1,
                      pad=0.02)
    cb.set_label(label='TEC (TECU)', size=18)

    map.figure.axes[0].tick_params(axis="both", labelsize=18)
    map.figure.axes[1].tick_params(axis="y", labelsize=18)

    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.11, right=1,
                        hspace=0.0, wspace=0.0)

    fig.savefig(output_file, format='png', dpi=100)


def plot_tec_map_raster(tec_map, tec_min=0, tec_max=160,
                        output_file='tec_map.png'):

    # grid_lon, grid_lat = np.meshgrid(lon, lat)

    fig = Figure(figsize=(4.9, 4.9))

    ax = fig.subplots(1, subplot_kw=dict(
        projection=ccrs.PlateCarree()))


    cmap = mpl.colormaps.get_cmap("jet").copy()
    cmap.set_bad(alpha=0)

    ax.imshow(tec_map,
              vmin=float(tec_min),
              vmax=float(tec_max),
              cmap=cmap,
              alpha=1,
              aspect='equal',
              interpolation='none')
    ax.set_axis_off()

    fig.subplots_adjust(top=1, bottom=0, left=0, right=1,
                        hspace=0.0, wspace=0.0)

    fig.savefig(output_file, format='png', dpi=100, transparent=True)