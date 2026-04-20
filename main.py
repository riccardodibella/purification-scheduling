# pyright: basic
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# rng = np.random.default_rng(0)
rng = np.random.default_rng(0)

def single_pair_greedy_policy_highest(l: list[float]) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0], reverse=True)
    return [(working_l[0][1],working_l[1][1])]

def single_pair_greedy_policy_lowest(l: list[float]) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0], reverse=False)
    return [(working_l[0][1],working_l[1][1])]

def single_pair_random_policy(l: list[float]) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    i, j = rng.choice(len(l), size=2, replace=False)
    return [(int(i), int(j))]


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

def sample_bernoulli(ok_prob: float) -> bool:
    assert ok_prob >= 0
    assert ok_prob <= 1
    return rng.random() < ok_prob

def check_feasible_schedule(choices: list[tuple[int, int]]) -> bool:
    # we don't check that all the choices are made within the length of the list
    # we just check that choices don't overlap, and therefore that no pair of choices have a qubit in common
    
    count_dict = {}
    for two_qubits_choice in choices:
        for qubit_index in two_qubits_choice:
            count_dict[qubit_index] = count_dict.get(qubit_index, 0) + 1

    for k in count_dict.keys():
        if count_dict[k] > 1:
            return False
    return True

def purify_sample(fidelities: list[float], choices: list[tuple[int, int]]) -> list[float]:
    assert check_feasible_schedule(choices)
    
    fids = fidelities.copy() # shallow copy (it's fine because we have only floats)
    new_pairs_fids = []
    for c in choices:
        purif_ok = sample_bernoulli(bit_flip_channel_purif_ok_prob(fids[c[0]], fids[c[1]]))
        if purif_ok:
            new_pairs_fids += [bit_flip_channel_purif_res_fidelity(fids[c[0]], fids[c[1]])]
        fids[c[0]] = -1
        fids[c[1]] = -1
    
    fids = [f for f in fids if f >= 0] # filter out the -1s
    return fids + new_pairs_fids

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

def run_randomized_simulation(iter_list: None | list[float] = None) -> int:
    if iter_list is None:
        iter_list = [rng.uniform(0.5, 0.9) for _ in range(100)]
        # iter_list = [0.7 for _ in range(100)]
    usable_list = []
    fidelity_threshold = 0.9
    while len(iter_list) > 1:
        choice = single_pair_random_policy(iter_list)
        iter_list = purify_sample(iter_list, choice)
        iter_list, above = filter_pairs_above_threshold(iter_list, fidelity_threshold)
        usable_list += above
    return len(usable_list)

def average_usable_pairs(results: dict[int, int]) -> float: 
    total_samples = sum(results.values())
    assert total_samples > 0
    weighted_sum = sum(outcome * count for outcome, count in results.items())
    
    return weighted_sum / total_samples

def plot_distribution_dict(results):
    # Plotting
    outcomes = sorted(results.keys())
    counts = [results[o] for o in outcomes]
    plt.bar(outcomes, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Usable Pairs')
    plt.ylabel('Frequency')
    plt.title('Outcome Distribution of Purification Simulation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    results = {}
    for i in range(5000):
        outcome = run_randomized_simulation()
        results[outcome] = results.get(outcome, 0) + 1
    print(average_usable_pairs(results))
    
    
    