#!/bin/bash

watch "ls state-210x160-$1 | wc -l && tmux ls && bash run_distributed_collect.sh"