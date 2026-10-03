# Near-critical constructions

`python verify_critical_capacity.py` checks the explicit proof construction.
`python two_look_search.py` searches a declared five-point grid of first-look
fractions and recomputes the best epsilon=1e-6 rule at high precision.
These are feasible lower bounds, not the exact continuous-time capacity solver.
At epsilon=1e-6 the latter family achieves about 0.3862 information-time units;
the simple construction used in the proof has a different, smaller value.
Neither is market P&L. Dependencies are listed in requirements.txt.
