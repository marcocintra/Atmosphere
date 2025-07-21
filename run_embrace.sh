#!/bin/bash

DATA_DIR=data/
INTERP_DIR=output/

DATASETS=(
'TF1_EMBRACE_TEC_maps_0800',
'TF1_EMBRACE_TEC_maps_1600',
'TF1_EMBRACE_TEC_maps_2000_2200_0000_0200_0400',
'TF3_EMBRACE_TEC_maps_0800',
'TF3_EMBRACE_TEC_maps_1600',
'TF3_EMBRACE_TEC_maps_2000_2200_0000_0200_0400'
)

for dataset in "${DATASETS[@]}"
do
  echo "Plotting original TEC maps for dataset: ${dataset}..."
  python 01_parallel_npy_plot.py "${DATA_DIR}${dataset}.npy"
  echo
done

echo
echo
for dataset in "${DATASETS[@]}"
do
  echo "Interpolating TEC maps for dataset: ${dataset}..."
  python 02_parallel_npy_interpolation.py "${DATA_DIR}${dataset}.npy"
  echo
done

echo
echo
for dataset in "${DATASETS[@]}"
do
  echo "Plotting interpolated TEC maps for dataset: ${dataset}..."
  python 03_parallel_interpolated_npy_plot.py "${INTERP_DIR}${dataset}_interp"
  echo
done
