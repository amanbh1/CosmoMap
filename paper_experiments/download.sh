#!/bin/bash
# Download the CAMELS Multifield Dataset 3D grids used in the paper into ~/camels.
#
#   bash download.sh                          # hydrodynamical grids at z = 0
#   ZS="0.0 0.5 1.0 1.5 2.0" bash download.sh # all five redshifts (about 34 GB each)
#   bash download.sh nbody                    # gravity-only total-matter grid at z = 0
#
# The location below is the one given in the Multifield Dataset documentation
# (https://camels-multifield-dataset.readthedocs.io); check there if it has moved.
set -e
CMD="https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/3D_grids/data"
mkdir -p ~/camels && cd ~/camels
if [ "${1:-}" = "nbody" ]; then
  wget -nv -c "$CMD/Nbody/Grids_Mtot_Nbody_IllustrisTNG_LH_128_z=0.0.npy"
else
  wget -nv -c "$CMD/IllustrisTNG/params_LH_IllustrisTNG.txt"
  for z in ${ZS:-0.0}; do
    for f in Mcdm Vcdm T ne; do wget -nv -c "$CMD/IllustrisTNG/Grids_${f}_IllustrisTNG_LH_128_z=${z}.npy"; done
  done
fi
echo "done: $(ls ~/camels | wc -l) files in ~/camels"
