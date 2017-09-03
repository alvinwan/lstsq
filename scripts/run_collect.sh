#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Need name of game!"
    exit
fi

source $ENV3 && \
python train-atari.py --task play --load /data/alvin/models/$1.tfmodel --env $1 --N_p=1000
