from typing import Tuple, List

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import scale
from numpy import pi, sqrt
import numpy as np
from matplotlib import pyplot as plt

modes = 8
prog = sf.Program(modes)


def ns_gate(q, input_mode: int, ancilla_modes: Tuple[int, int]):
    anc_mode_1, anc_mode_2 = ancilla_modes

    Fock(1) | q[anc_mode_1]
    Fock(0) | q[anc_mode_2]

    # Zgate(pi)               | q[input_mode]
    BSgate(pi/8, 0)         | (q[anc_mode_1], q[anc_mode_2])
    BSgate(0.3640567*pi, 0) | (q[input_mode], q[anc_mode_1])
    BSgate(-pi/8, 0)        | (q[anc_mode_1], q[anc_mode_2])

    MeasureFock(select=1) | q[anc_mode_1]
    MeasureFock(select=0) | q[anc_mode_2]


# with prog.context as q:
#     # prepare initial states
#     Fock(1) | q[2]
#     Vacuum() | q[1]
#     Vacuum() | q[0]
#
#     # apply gates
#     BS = BSgate(pi/4, pi)
#     BS | (q[1], q[0])
#     BS | (q[2], q[1])
#
#     # Perform homodyne measurements
#     MeasureX | q[2]
#     MeasureP | q[1]
#
#     # Displacement gates conditioned on
#     # the measurements
#     Xgate(scale(q[2], sqrt(0))) | q[0]
#     Zgate(custom(q[1])) | q[0]


def csgn_gate(q, control_modes: Tuple[int, int], target_modes: Tuple[int, int],
              anc_modes=List[int], ns_anc_modes=List[Tuple[int, int]]):
    c0, c1 = control_modes
    t0, t1 = target_modes
    ns_anc_modes0, ns_anc_modes1 = ns_anc_modes

    BSgate(theta=pi/4, phi=0)   | (q[t0], q[t1])
    BSgate(theta=pi/4, phi=0)   | (q[t0], q[c1])

    ns_gate(q, c1, ns_anc_modes0)
    ns_gate(q, t0, ns_anc_modes1)

    BSgate(theta=pi/4, phi=0)   | (q[t0], q[c1])
    BSgate(theta=pi/4, phi=0)   | (q[t0], q[t1])

    MeasureX | q[c0]
    MeasureX | q[c1]
    MeasureX | q[t0]
    MeasureX | q[t1]


def prep_cnot(q, control, target, control_modes, target_modes, anc_modes):
    if control:
        Fock(0) | q[control_modes[0]]
        Fock(1) | q[control_modes[1]]
    else:
        Fock(1) | q[control_modes[0]]
        Fock(0) | q[control_modes[1]]

    if target:
        Fock(0) | q[target_modes[0]]
        Fock(1) | q[target_modes[1]]
    else:
        Fock(1) | q[target_modes[0]]
        Fock(0) | q[target_modes[1]]


control = (0, 1)
target = (2, 3)
ns_ancs = [(4, 5), (6, 7)]
cnot_ancs = []
with prog.context as q:
    prep_cnot(q, False, False, control, target, ns_ancs)
    csgn_gate(q, control, target, cnot_ancs, ns_ancs)
    # ns_gate(q, 0, (1, 2))

# print(prog.draw_circuit())


prog.print()

dims = 2
runs = 10
eng = sf.Engine('fock', backend_options={'cutoff_dim': dims})

acc_prob_modes = None
samples = np.zeros((runs, modes))
print(samples.shape)
for i in range(runs):
    result = eng.run(prog, run_options={"shots": 1, "modes": [0, 1, 2, 3]}, compile_options={"warn_connected": False})
    state = result.state
    fock_probs = state.all_fock_probs()
    # print("SAMP:", result.samples)
    # print("FOCK PROBS:", fock_probs)
    for j in range(modes):
        samples[i, j] = result.samples[j]
    if acc_prob_modes is None:
        acc_prob_modes = fock_probs
    else:
        acc_prob_modes = np.add(fock_probs, acc_prob_modes)
    samp_avg = np.sum(samples, axis=0) / (i + 1)
    print(i, "SAMP SUM: ", samp_avg)

fock_probs = state.all_fock_probs()
print(fock_probs.shape)

print(fock_probs)
probs_mode0 = np.sum(fock_probs, axis=0)

print(acc_prob_modes[:dims] / runs)

for i in range(4):
    print("Mode {}:\n".format(i), np.sum(fock_probs, axis=i))

print(samples)
accepted = 0
samp_avg = np.sum(samples, axis=0) / runs
print("SAMP_AVG:", samp_avg)

plt.bar(range(dims), acc_prob_modes[:dims] / runs)
plt.xlabel('Fock state')
plt.ylabel('Marginal probability')
plt.title('Mode 0')
plt.show()

