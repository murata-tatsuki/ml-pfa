"""Single-particle timing cut; skip events with >= 2 input truth_particles.

Usage: python timingcut_single_particle.py -i input.h5 -o output.h5
Requires the full input H5 containing truth_particles. Events with zero truth
particles remain eligible. Output is a compact training H5, as in timingcut.py.
"""
from timingcut import main


if __name__ == '__main__':
    main(single_particle=True)
