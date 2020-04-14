# Optimization Process
The following is an example of the steps take by the optomiazation procedure.


## Step 1: Input from user.
Input program:
```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[2],q[3];
cx q[4],q[0];
h q[3];
```

Visualized as:
```
             ┌───┐     
q_0: |0>─────┤ X ├─────
             └─┬─┘     
q_1: |0>───────┼───────
               │       
q_2: |0>──■────┼───────
        ┌─┴─┐  │  ┌───┐
q_3: |0>┤ X ├──┼──┤ H ├
        └───┘  │  └───┘
q_4: |0>───────■───────
                       
```

Input coupling graph:
```
0,1
1,0
1,2
2,1
2,3
3,2
3,4
4,3
```

Input parameters:
```
max_circuit_time = 4
max_swaps_addable = 1
```

## Step 2: Remap input circuit register.
The qubit register is remapped so that qubits are consectuvitely ordered for ease of modeling.
```
               ┌───┐     
(q0) q0|0>─────┤ X ├─────
               └─┬─┘     
(q2) q1|0>──■────┼───────
          ┌─┴─┐  │  ┌───┐
(q3) q2|0>┤ X ├──┼──┤ H ├
          └───┘  │  └───┘
(q4) q3|0>───────■───────
                         
(q1) q4|0>───────────────
```

We hold onto this remapping so that the eventual running of the optimized program succeeds.

## Step 3: Convert circuit to DAG.
Visualized DAG:

Topological ordering of gate nodes:


## Step 4: Generate variables and constraints based on each node in circuit.
At each optimization round we set the optimization criteria, by fixing them:
### Step 4a: We set a value for the criteria as follows:

constrain_circuit_end_time:
```
[circuit_end_time <= 6,
 circuit_end_time >=
 gate_start_time__2_0 + gate_duration__2_0,
 circuit_end_time >=
 gate_start_time__2_1 + gate_duration__2_1,
 circuit_end_time >=
 gate_start_time__2_2 + gate_duration__2_2,
 circuit_end_time >=
 gate_start_time__2_3 + gate_duration__2_3,
 circuit_end_time >=
 gate_start_time__2_4 + gate_duration__2_4]
```

constrain_swaps_added:
```
[swaps_added >= 0,
 swaps_added == 1,
 AtMost((swap_gate_insertion__0_0_0_1,
         swap_gate_insertion__0_0_1_2,
         swap_gate_insertion__0_0_2_3,
         swap_gate_insertion__0_0_3_4,
         swap_gate_insertion__0_1_0_1,
         swap_gate_insertion__0_1_1_2,
         swap_gate_insertion__0_1_2_3,
         swap_gate_insertion__0_1_3_4,
         swap_gate_insertion__0_2_0_1,
         swap_gate_insertion__0_2_1_2,
         swap_gate_insertion__0_2_2_3,
         swap_gate_insertion__0_2_3_4,
         swap_gate_insertion__0_3_0_1,
         swap_gate_insertion__0_3_1_2,
         swap_gate_insertion__0_3_2_3,
         swap_gate_insertion__0_3_3_4,
         swap_gate_insertion__1_0_0_1,
         swap_gate_insertion__1_0_1_2,
         swap_gate_insertion__1_0_2_3,
         swap_gate_insertion__1_0_3_4,
         swap_gate_insertion__1_1_0_1,
         swap_gate_insertion__1_1_1_2,
         swap_gate_insertion__1_1_2_3,
         swap_gate_insertion__1_1_3_4,
         swap_gate_insertion__1_2_0_1,
         swap_gate_insertion__1_2_1_2,
         swap_gate_insertion__1_2_2_3,
         swap_gate_insertion__1_2_3_4,
         swap_gate_insertion__1_3_0_1,
         swap_gate_insertion__1_3_1_2,
         swap_gate_insertion__1_3_2_3,
         swap_gate_insertion__1_3_3_4,
         swap_gate_insertion__2_0_0_1,
         swap_gate_insertion__2_0_1_2,
         swap_gate_insertion__2_0_2_3,
         swap_gate_insertion__2_0_3_4,
         swap_gate_insertion__2_1_0_1,
         swap_gate_insertion__2_1_1_2,
         swap_gate_insertion__2_1_2_3,
         swap_gate_insertion__2_1_3_4,
         swap_gate_insertion__2_2_0_1,
         swap_gate_insertion__2_2_1_2,
         swap_gate_insertion__2_2_2_3,
         swap_gate_insertion__2_2_3_4,
         swap_gate_insertion__2_3_0_1,
         swap_gate_insertion__2_3_1_2,
         swap_gate_insertion__2_3_2_3,
         swap_gate_insertion__2_3_3_4),
        1)]
```

### Step 4b:
A sampling of contraints looks like the following:

constrain_swap_qubit_mapping:
```
swap_qubit_mapping__0_0_0 < 4,
 swap_qubit_mapping__0_0_1 >= -1,
 swap_qubit_mapping__0_0_1 < 4,
 swap_qubit_mapping__0_0_2 >= -1,
 swap_qubit_mapping__0_0_2 < 4,
 swap_qubit_mapping__0_0_3 >= -1,
 swap_qubit_mapping__0_0_3 < 4,
 swap_qubit_mapping__0_0_4 >= -1,
 swap_qubit_mapping__0_0_4 < 4,
 Distinct(swap_qubit_mapping__0_0_0,
          swap_qubit_mapping__0_0_1,
          swap_qubit_mapping__0_0_2,
          swap_qubit_mapping__0_0_3,
          swap_qubit_mapping__0_0_4),
 Implies(swap_gate_insertion__0_0_0_1,
         And(input_qubit_mapping__0 ==
             swap_qubit_mapping__0_0_1,
             input_qubit_mapping__1 ==
             swap_qubit_mapping__0_0_0)),
...
```

constrain_gate_ends_before_end_time:
```
gate_start_time__0_0 + gate_duration__0_0 >= 0,
 gate_start_time__0_0 + gate_duration__0_0 < 6,
 gate_start_time__0_1 + gate_duration__0_1 >= 0,
 gate_start_time__0_1 + gate_duration__0_1 < 6,
 gate_start_time__0_2 + gate_duration__0_2 >= 0,
 gate_start_time__0_2 + gate_duration__0_2 < 6,
 gate_start_time__0_3 + gate_duration__0_3 >= 0,
 gate_start_time__0_3 + gate_duration__0_3 < 6,
 gate_start_time__0_4 + gate_duration__0_4 >= 0,
 gate_start_time__0_4 + gate_duration__0_4 < 6,
 Implies(swap_gate_insertion__0_0_0_1,
         gate_start_time__0_0 == gate_start_time__0_1),
 Implies(swap_gate_insertion__0_0_1_2,
         gate_start_time__0_1 == gate_start_time__0_2),
 Implies(swap_gate_insertion__0_0_2_3,
         gate_start_time__0_2 == gate_start_time__0_3),
 Implies(swap_gate_insertion__0_0_3_4,
         gate_start_time__0_3 == gate_start_time__0_4),
 Implies(swap_gate_insertion__0_1_0_1,
         gate_start_time__0_0 == gate_start_time__0_1),
 Implies(swap_gate_insertion__0_1_1_2,
         gate_start_time__0_1 == gate_start_time__0_2),
 Implies(swap_gate_insertion__0_1_2_3,
         gate_start_time__0_2 == gate_start_time__0_3),
 Implies(swap_gate_insertion__0_1_3_4,
         gate_start_time__0_3 == gate_start_time__0_4),
 Implies(swap_gate_insertion__0_2_0_1,
         gate_start_time__0_0 == gate_start_time__0_1),
...
```

constrain_gate_duration:
```
gate_duration__0_0 >= swap_gate_duration__0_0,
 gate_duration__0_1 >= swap_gate_duration__0_1,
 gate_duration__0_2 >= swap_gate_duration__0_2,
 gate_duration__0_3 >= swap_gate_duration__0_3,
 gate_duration__0_4 >= swap_gate_duration__0_4,
 Implies(gate_qubit_mapping__0_0 == 0,
         gate_duration__0_0 >= 1 + swap_gate_duration__0_0),
 Implies(gate_qubit_mapping__0_1 == 0,
         gate_duration__0_1 >= 1 + swap_gate_duration__0_1),
 Implies(gate_qubit_mapping__0_2 == 0,
         gate_duration__0_2 >= 1 + swap_gate_duration__0_2),
 Implies(gate_qubit_mapping__0_3 == 0,
         gate_duration__0_3 >= 1 + swap_gate_duration__0_3),
 Implies(gate_qubit_mapping__0_4 == 0,
         gate_duration__0_4 >= 1 + swap_gate_duration__0_4),
 gate_duration__1_0 >= swap_gate_duration__1_0,
 gate_duration__1_1 >= swap_gate_duration__1_1,
 gate_duration__1_2 >= swap_gate_duration__1_2,
 gate_duration__1_3 >= swap_gate_duration__1_3,
 gate_duration__1_4 >= swap_gate_duration__1_4,
 Implies(And(gate_qubit_mapping__1_0 == 1,
             gate_qubit_mapping__1_1 == 2),
         And(And(gate_duration__1_0 >=
                 1 + swap_gate_duration__1_0,
                 gate_duration__1_1 >=
                 1 + swap_gate_duration__1_1,
                 gate_duration__1_0 >=
                 1 + swap_gate_duration__1_1,
                 gate_duration__1_1 >=
                 1 + swap_gate_duration__1_0),
             And(gate_start_time__1_0 + gate_duration__1_0 ==
                 gate_start_time__1_1 + gate_duration__1_1))),
 Implies(And(gate_qubit_mapping__1_0 == 1,
```

## Step 6: Convert to CNF and split clauses to a set size.
This means we have to convert all of the integer variables from above to CNF via bit vectors and bounds checking. 
```
...
 (or (not k!6601) k!6598)
  (or k!1106 k!6602)
  (or k!1106 k!5648)
  (or (not k!6603) k!6598)
  (or (not k!1107) k!6604)
  (or k!1107 k!6605)
  (or k!1107 k!6606)
  (or (not k!6607) k!6598)
  (or (not k!1108) k!6604)
  (or k!1108 k!6608)
  (or k!1108 k!6609)
  (or (not k!6610) k!6598)
  (or (not k!1109) (not k!6604))
  (or (not k!1109) (not k!1361))
  (or (not k!1109) (not k!6611))
  (or (not k!6612) k!6598)
  (or (not k!6598) (not swap_gate_insertion__2_1_2_3))
  (or (not k!6598) (not swap_gate_insertion__2_2_2_3))
  (or (not k!6598) (not swap_gate_insertion__2_3_2_3))
  (or (not k!6598) (not swap_gate_insertion__2_4_2_3))
  (or (not k!6598) (not swap_gate_insertion__2_5_2_3))
  (or (not swap_gate_insertion__2_0_3_4) (not k!6613))
  (or (not swap_gate_insertion__2_0_3_4) (not k!6614))
  (or (not swap_gate_insertion__2_0_3_4) (not swap_gate_insertion__2_0_4_5))
  (or (not swap_gate_insertion__2_0_3_4) (not swap_gate_insertion__2_0_5_6))
  (or (not swap_gate_insertion__2_0_3_4) dsort!599)
  (or swap_gate_insertion__2_0_3_4 k!6600)
  (or swap_gate_insertion__2_0_3_4 k!6615)
...
```

## Step 7: Run on Solver:
We pass the CNF to the quantum or classical solver, who transform the CNF to an internal form. (Think a massive Grover's oracle.)
We get back the assignment of truth values for the boolean values:

```
  ...
  swap_gate_insertion__2_2_3_4 = T
  swap_gate_insertion__2_3_2_3 = F
  swap_gate_insertion__2_3_3_4 = F
  k!3510 = T
  k!3512 = F
  k!3513 = F 
  k!3514 = T
  ...
```
We can now reconstruct the original integer variables:
```
    ...
    circuit_end_time = BitVec2Int([k!3510, k!3512, k!3513, k!3514])
    ...
```

Because we set the optimazation criteria in the constraints, if we get a satisifiable result, we can either continue the optimization process by setting new criteria values in step 4a and continuing in step 5. Otherwise we can proceed to the next step.
 
### Step 8: Reconstruct circuit.
We take those same reinterpreted values and construct the circuit from the assignment of variables. 
 
```
q_0: |0>────────────■──
                    │  
q_1: |0>──■─────────┼──
        ┌─┴─┐┌───┐  │  
q_2: |0>┤ X ├┤ H ├──┼──
        └───┘└───┘┌─┴─┐
q_3: |0>──X───────┤ X ├
          │       └───┘
q_4: |0>──X────────────
```                    
 

