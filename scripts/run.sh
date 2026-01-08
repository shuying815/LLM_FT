#!/bin/bash

if [ "$1" = "lora" ]; then
    python ./src/lora.py
elif [ "$1" = "prefix" ]; then
    python ./src/prefix.py
elif [ "$1" = "adapter" ]; then
    python ./src/adapter.py
elif [ "$1" = "dpo" ]; then
    python ./src/dpo.py
elif [ "$1" = "dpo_optimize" ]; then
    python ./src/dpo_optimize.py
else
    echo "用法: $0 [lora | prefix | adapter | dpo | dpo_optimize]"
    exit 1
fi