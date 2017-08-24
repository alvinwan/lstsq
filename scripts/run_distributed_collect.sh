#!/usr/bin/env bash

for i in `seq 0 20`
do
    tmux new-session -d -s "collect_$i" /data/alvin/lstsq/run_collect.sh
done
tmux ls