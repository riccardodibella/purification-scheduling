# pyright: basic
from itertools import product
from typing import Callable
from line_profiler import profile

import numpy as np

rng = np.random.default_rng(0)

PolicyFunction = Callable[[list[float], float], list[tuple[int, int]]]

def single_pair_greedy_policy_highest(l: list[float], thresh: float) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0], reverse=True)
    return [(working_l[0][1],working_l[1][1])]

def single_pair_greedy_policy_lowest(l: list[float], thresh: float) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0], reverse=False)
    return [(working_l[0][1],working_l[1][1])]

def all_pairs_policy_opposite(l: list[float], thresh: float) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0], reverse=True)
    pairs: list[tuple[int, int]] = []
    for i in range(0, int(len(working_l)/2)):
        idx1 = working_l[i][1]
        idx2 = working_l[len(working_l)-1-i][1]
        pairs += [(idx1, idx2)]
    return pairs



def bit_flip_channel_purif_ok_prob(fid1: float, fid2: float) -> float:
    assert fid1 >= 0
    assert fid1 <= 1
    assert fid2 >= 0
    assert fid2 <= 1
    return fid1 * fid2 + (1 - fid1) * (1 - fid2)

def bit_flip_channel_purif_res_fidelity(fid1: float, fid2: float) -> float:
    assert fid1 >= 0
    assert fid1 <= 1
    assert fid2 >= 0
    assert fid2 <= 1
    return  fid1 * fid2  / ( fid1 * fid2 + (1 - fid1) * (1 - fid2) )

def werner_channel_purif_ok_prob(fid1: float, fid2: float) -> float:
    assert fid1 >= 0
    assert fid1 <= 1
    assert fid2 >= 0
    assert fid2 <= 1
    return fid1 * fid2 + (1/3) * (fid1 + fid2 - 2 * fid1 * fid2) + (5/9) * (1 - fid1) * (1 - fid2)

def werner_channel_purif_res_fidelity(fid1: float, fid2: float) -> float:
    assert fid1 >= 0
    assert fid1 <= 1
    assert fid2 >= 0
    assert fid2 <= 1
    return  ( fid1 * fid2 + (1/9) * (1 - fid1) * (1 - fid2) ) / ( fid1 * fid2 + (1/3) * (fid1 + fid2 - 2 * fid1 * fid2) + (5/9) * (1 - fid1) * (1 - fid2) )

def check_feasible_schedule(choices: list[tuple[int, int]]) -> bool:
    # we don't check that all the choices are made within the length of the list
    # we just check that choices don't overlap, and therefore that no pair of choices have a qubit in common
    
    count_dict: dict[int, int] = {}
    for two_qubits_choice in choices:
        for qubit_index in two_qubits_choice:
            count_dict[qubit_index] = count_dict.get(qubit_index, 0) + 1

    for k in count_dict.keys():
        if count_dict[k] > 1:
            return False
    return True

# the first list is the list of fidelities below threshold, the second one are the fidelities above
def filter_pairs_above_threshold(fidelities: list[float], threshold: float) -> tuple[list[float], list[float]]:
    below: list[float] = []
    above: list[float] = []
    for f in fidelities:
        if f >= threshold:
            above += [f]
        else:
            below += [f]
    return below, above

def average_usable_pairs_from_distribution(distribution: list[tuple[float, tuple[int, int, list[float]]]]) -> float: 
    ret = 0.0
    for entry in distribution:
        prob = entry[0]
        usable = entry[1][0]
        ret += prob * usable
    return ret

def filter_usable_pairs(pairs: list[float], threshold: float) -> tuple[int, list[float]]:
    remaining_pairs = [p for p in pairs if p < threshold]
    usable_counter = len(pairs) - len(remaining_pairs)
    return usable_counter, remaining_pairs

def bitstrings(n: int):
    return [list(bits) for bits in product([False, True], repeat=n)]

@profile
def exact_recursive_simulation(policy: PolicyFunction, input_fidelities: list[float], previous_iterations: int = 0) -> list[tuple[float, tuple[int, int, list[float]]]]:
    """
    Return type: [(probability, (# of usable pairs, # of iterations, [residual fidelities]))]
    """
    fidelity_threshold = 0.98

    if(len(input_fidelities) < 2):
        return [(1, (0, previous_iterations+1, input_fidelities))]

    list_after_current_step: list[tuple[float, tuple[int, int, list[float]]]] = []
    choices = policy(input_fidelities, fidelity_threshold)
    assert check_feasible_schedule(choices)
    choices_ok_probabilities = [bit_flip_channel_purif_ok_prob(input_fidelities[c[0]], input_fidelities[c[1]]) for c in choices]
    choices_res_fidelities = [bit_flip_channel_purif_res_fidelity(input_fidelities[c[0]], input_fidelities[c[1]]) for c in choices]
    bss = bitstrings(len(choices))
    for outcome_i in range(2**len(choices)):
        outcome_bitstring = bss[outcome_i]

        # Calculation of outcome probability
        outcome_probability = 1.0
        for choice_i in range(len(choices)):
            choice_outcome = outcome_bitstring[choice_i]
            outcome_ok_probability = choices_ok_probabilities[choice_i]
            outcome_probability *= outcome_ok_probability if choice_outcome is True else (1.0 - outcome_ok_probability)

        # Calculation of resulting fidelities list (before usable pairs filtering)
        outcome_fidelities = input_fidelities.copy()
        new_fidelities: list[float] = []
        for choice_i in range(len(choices)):
            c = choices[choice_i]
            choice_outcome = outcome_bitstring[choice_i]
            if choice_outcome is True:
                new_fidelities += [choices_res_fidelities[choice_i]]
            outcome_fidelities[c[0]] = -1
            outcome_fidelities[c[1]] = -1
        outcome_fidelities = [f for f in outcome_fidelities if f >= 0] # filter out the -1s
        outcome_fidelities += new_fidelities

        # Filter usable pairs based on the fidelity threshold
        outcome_usable_pairs, outcome_filtered_fidelities = filter_usable_pairs(outcome_fidelities, fidelity_threshold)
        
        list_after_current_step += [(outcome_probability, (outcome_usable_pairs, previous_iterations+1, outcome_filtered_fidelities))]

    list_after_recursion: list[tuple[float, tuple[int, int, list[float]]]] = []
    for current_outcome_prob, (current_outcome_usable, current_outcome_iter, current_outcome_remaining_fids) in list_after_current_step:
        recursion_results = exact_recursive_simulation(policy, current_outcome_remaining_fids, current_outcome_iter)
        for res_prob, (res_usable, res_iter, res_remaining_fids) in recursion_results:
            new_entry = (
                    current_outcome_prob * res_prob,
                (
                    current_outcome_usable + res_usable,
                    res_iter,
                    res_remaining_fids
                )
            )
            list_after_recursion.append(new_entry)
    return list_after_recursion

if __name__ == "__main__":
    input_fid_list = [rng.uniform(0.5, 0.9) for _ in range(20)]
    for policy in [single_pair_greedy_policy_highest]:
        end_distribution = exact_recursive_simulation(policy, input_fid_list)
        print(f"{policy.__name__}: {average_usable_pairs_from_distribution(end_distribution)}")
    
    
    