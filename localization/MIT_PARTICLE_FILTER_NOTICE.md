# MIT RACECAR particle-filter provenance

The implementation in `mit_particle_filter.py` is a Python 3.11, ROS-free
port of the Monte Carlo localization algorithm published in:

- <https://github.com/mit-racecar/particle_filter>
- Source commit: `95613c656f32584143aa98e0f950f1be55f0c6dd`
- Package license declared by the upstream ROS package: MIT

The official F1/10 reference manual recommends this package for particle
filter localization. Its motion model, beam sensor model, initialization, and
launch-file defaults are retained here. ROS1 message transport and RangeLibc
are replaced by NumPy inputs and F1TENTH Gym's distance-transform ray-marching
method so the experiment can run in the repository's Python 3.11 environment.
