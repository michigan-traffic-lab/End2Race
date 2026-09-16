#!/usr/bin/env bash
set -e

pip install ./f1tenth_sim/f1tenth_gym
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install quadprog==0.1.13 --no-deps
# gym 0.23.1 depends on gym_notices, which prints a deprecation notice on every import
pip uninstall -y gym_notices
