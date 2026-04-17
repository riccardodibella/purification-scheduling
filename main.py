# pyright: basic
import numpy as np

rng = np.random.default_rng(0)

# In general, some other parameters may be useful: iteration_number: int, ready_pairs_count: int
# We ignore those for now, since we don't use them
def single_pair_greedy_policy(l: list[float]) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0], reverse=True) # descending order (higher fidelity first)
    return [(working_l[0][1],working_l[1][1])]


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

if __name__ == "__main__":
    iter_list = [rng.uniform(0.5, 0.9) for _ in range(100)]
    usable_list = []
    fidelity_threshold = 0.9
    while len(iter_list) > 1:
        choice = single_pair_greedy_policy(iter_list)
        iter_list = purify_sample(iter_list, choice)
        iter_list, above = filter_pairs_above_threshold(iter_list, fidelity_threshold)
        usable_list += above
    
    print(len(usable_list))
    
    