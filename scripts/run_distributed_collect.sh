#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Need name of game!"
    exit
fi

for i in `seq 0 4`
do
    tmux new-session -d -s "collect_$1_$i" /data/alvin/lstsq/scripts/run_collect.sh $1
done
watch tmux ls
