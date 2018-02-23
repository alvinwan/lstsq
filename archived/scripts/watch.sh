#!/bin/bash

watch "ls state-210x160-$1 | wc -l && tmux ls && bash scripts/run_distributed_collect.sh $1"