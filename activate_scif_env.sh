#!/bin/bash -l
eval "$(mamba shell hook --shell bash)"
mamba activate SCIF_env

python PrintEnv.py