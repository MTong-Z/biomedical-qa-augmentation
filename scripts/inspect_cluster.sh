#!/usr/bin/env bash
set -euo pipefail
hostname
if command -v sinfo >/dev/null 2>&1; then
    sinfo -h -o '%P|%a|%l|%G'
fi
if command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr -n -P show assoc where "user=$USER" format=Account,Partition,QOS
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
