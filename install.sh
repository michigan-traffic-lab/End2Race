#!/usr/bin/env bash
set -e

python -m pip install ./f1tenth_sim/f1tenth_gym
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install quadprog==0.1.13 --no-deps
# gym 0.23.1 depends on gym_notices, which prints a deprecation notice on every import
python -m pip uninstall -y gym_notices
