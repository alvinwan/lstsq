#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Need name of game!"
    exit
fi

source $ENV3 && \
python collect.py $1
#python /data/alvin/lstsq/train-atari.py --task play --load /data/alvin/models/$1.npy --env $1 --N_p=10000
