#!/bin/bash

DATA_DIR=data/
INTERP_DIR=output/

DATASETS=(
'TF1_MAGGIA_TEC_maps_0800'
'TF1_MAGGIA_TEC_maps_1600'
'TF1_MAGGIA_TEC_maps_2000_2200_0000_0200_0400'
#'TF2_MAGGIA_TEC_maps_0800'
#'TF2_MAGGIA_TEC_maps_1600'
#'TF2_MAGGIA_TEC_maps_2000_2200_0000_0200_0400'
'TF3_MAGGIA_TEC_maps_0800_30m'
'TF3_MAGGIA_TEC_maps_1600_30m'
'TF3_MAGGIA_TEC_maps_2000_0400_30m'
)

for dataset in "${DATASETS[@]}"
do
  echo "Plotting original TEC maps for dataset: ${dataset}..."
  python 01_serial_npy_plot.py "${DATA_DIR}${dataset}.npy"
  echo
done

echo
echo
for dataset in "${DATASETS[@]}"
do
  echo "Interpolating TEC maps for dataset: ${dataset}..."
  python 02_serial_npy_interpolation.py "${DATA_DIR}${dataset}.npy"
  echo
done

echo
echo
for dataset in "${DATASETS[@]}"
do
  echo "Plotting interpolated TEC maps for dataset: ${dataset}..."
  python 03_serial_interpolated_npy_plot.py "${INTERP_DIR}${dataset}_interp"
  echo
done