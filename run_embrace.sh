#!/bin/bash

DATA_DIR=data/
INTERP_DIR=output/

DATASETS=(
mapas1_embrace_2022_2024_0800
mapas1_embrace_2022_2024_1600
mapas1_embrace_2022_2024_2000_2200_0000_0200_0400
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