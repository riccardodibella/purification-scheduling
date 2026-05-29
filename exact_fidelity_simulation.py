# pyright: strict
from dataclasses import dataclass
from itertools import chain, combinations, product
from typing import Callable
from math import log10, ceil
import numpy as np
from enum import Enum, auto

rng = np.random.default_rng(0)


PolicyFunction = Callable[[list[tuple[str, float]], float], list[tuple[int, int]]]

StateDescription = str
ChoiceDescription = str
lookup_dict: dict[StateDescription, ChoiceDescription] = {}

def sort_fid_named_list(l: list[tuple[str, float]], highestFirst: bool = True) -> list[tuple[str, float]]:
    return sorted(l, key=lambda x: x[1], reverse=highestFirst)

def sort_str_named_list(l: list[tuple[str, float]]) -> list[tuple[str, float]]:
    # Lexicographic ascending order
    return sorted(l, key=lambda x: x[0], reverse=False)

def encode_state_description(l: list[tuple[str, float]]) -> StateDescription:
    l = sort_str_named_list(l)
    return ','.join([t[0] for t in l])

def encode_purified_pair(st1: str, st2: str) -> str:
    return f"<{st1}+{st2}>"

def decode_choice_description(s: ChoiceDescription) -> list[tuple[str, str]]:
    arr: list[str] = s.split(",")
    if len(arr) == 0:
        return []
    if len(arr) == 1 and ":" not in s:
        return []
    return [(elem.split(":")[0], elem.split(":")[1]) for elem in arr]

def decode_choice(l: list[tuple[str, float]], choice: ChoiceDescription) -> list[tuple[int, int]]:
    qubit_names_list: list[tuple[str, str]]=decode_choice_description(choice)
    to_return: list[tuple[int, int]] = []
    for names_tuple in qubit_names_list:
        index0: int = -1
        index1: int = -1

        for index_iter, fid_tuple in enumerate(l):
            if fid_tuple[0] == names_tuple[0]:
                index0 = index_iter
            elif fid_tuple[0] == names_tuple[1]:
                index1 = index_iter
            if index0 >= 0 and index1 >= 0:
                break

        to_return += [(index0, index1)]
    return to_return

def lookup_policy(l: list[tuple[str, float]], thresh: float) -> list[tuple[int, int]]:
    input_state: StateDescription = encode_state_description(l)
    if input_state not in lookup_dict.keys():
        # Unexpected state! (not in dict)
        print("Piva piva l'olio d'oliva")
        print(input_state)
        print(lookup_dict.keys())
        assert False
    choice_str: ChoiceDescription = lookup_dict[input_state]
    to_return = decode_choice(l, choice_str)
    return to_return

def single_pair_greedy_policy_highest(l: list[tuple[str, float]], thresh: float) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=True)
    return [(working_l[0][1],working_l[1][1])]


def single_pair_greedy_policy_lowest(l: list[tuple[str, float]], thresh: float) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=False)
    return [(working_l[0][1],working_l[1][1])]

def all_pairs_policy_opposite(l: list[tuple[str, float]], thresh: float) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=True)
    pairs: list[tuple[int, int]] = []
    for i in range(0, int(len(working_l)/2)):
        idx1 = working_l[i][1]
        idx2 = working_l[len(working_l)-1-i][1]
        pairs += [(idx1, idx2)]
    return pairs


def gen_initial_named_pairs() -> list[tuple[str, float]]:
    fids: list[float] = gen_initial_pairs()
    fids = sorted(fids, reverse=True)
    num_chars = ceil(log10(len(fids)))
    to_return = [(f"{i}".zfill(num_chars), fids[i]) for i in range(len(fids))]
    return to_return



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

class PurificationModel(Enum):
    BIT_FLIP = auto(),
    WERNER = auto()

def purif_ok_prob(model: PurificationModel, fid1: float, fid2: float) -> float:
    if model == PurificationModel.BIT_FLIP:
        return bit_flip_channel_purif_ok_prob(fid1, fid2)
    elif model == PurificationModel.WERNER:
        return werner_channel_purif_ok_prob(fid1, fid2)
    raise NotImplementedError(f"Purification model {model} not supported (purify_ok_prob)")

def purif_res_fidelity(model: PurificationModel, fid1: float, fid2: float) -> float:
    if model == PurificationModel.BIT_FLIP:
        return bit_flip_channel_purif_res_fidelity(fid1, fid2)
    elif model == PurificationModel.WERNER:
        return werner_channel_purif_res_fidelity(fid1, fid2)
    raise NotImplementedError(f"Purification model {model} not supported (purify_ok_prob)")


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

def bitstrings(n: int):
    return [list(bits) for bits in product([False, True], repeat=n)]

def filter_usable_pairs(pairs: list[tuple[str, float]], threshold: float) -> tuple[int, list[tuple[str, float]]]:
    remaining_pairs = [p for p in pairs if p[1] < threshold]
    usable_counter = len(pairs) - len(remaining_pairs)
    return usable_counter, remaining_pairs

def gen_initial_pairs() -> list[float]:
    return [0.85, 0.8, 0.7, 0.6]

def generate_immediate_termination_lookup_dict(initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel):
    initial_state = encode_state_description(initial_fids)
    lookup_dict[initial_state] = ""

def str_powerset(keys: list[str])->chain[tuple[str, ...]]: # By default returns a lazy iterable, cast to list if you want all at once
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    return chain.from_iterable(combinations(keys, r) for r in range(len(keys)+1))

@dataclass
class WorkingDictEntry:
    action: str | None = None
    definitive: bool = False
    possible_actions: list[str] | None = None

def set_lookup_dict(working_dict: dict[str, WorkingDictEntry]):
    lookup_dict.clear()
    for k in working_dict.keys():
        action = working_dict[k].action
        if action is not None:
            lookup_dict[k] = action

def generate_possible_states(inputs: list[str]) -> list[str]:
    # full_state_set: set[str] = set()

    # possible_subsets = list(str_powerset(inputs))
    # for s in possible_subsets:

    return []

def generate_possible_actions(state_str: str) -> list[str]:
    return []

def generate_lookup_dict(initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel):
    generate_immediate_termination_lookup_dict(initial_fids, threshold, model)

    input_keys: list[str] = [f[0] for f in initial_fids]
    # possible_subsets = list(str_powerset(input_keys))

    possible_states = generate_possible_states(input_keys)

    working_dict: dict[str, WorkingDictEntry] = {}
    for state_string in possible_states:
        working_dict[state_string] = WorkingDictEntry(action=None, definitive=False, possible_actions=generate_possible_actions(state_string))
    return

def exact_recursive_simulation(policy: PolicyFunction, input_fidelities: list[tuple[str, float]], fidelity_threshold: float, model: PurificationModel, previous_iterations: int = 0) -> list[tuple[float, tuple[int, int, list[tuple[str, float]]]]]:
    """
    Return type: [(probability, (# of usable pairs, # of iterations, [(remaining_keys, remaining_fids)]))]
    """
    if(len(input_fidelities) < 2):
        return [(1, (0, previous_iterations+1, input_fidelities))]
    
    list_after_current_step: list[tuple[float, tuple[int, int, list[tuple[str, float]]]]] = []
    choices = policy(input_fidelities, fidelity_threshold)
    assert check_feasible_schedule(choices)

    if len(choices) == 0:
        # empty choice list means that the purification path ends here and leftover pairs stay unused
        return [(1, (0, previous_iterations+1, input_fidelities))]

    choices_ok_probabilities = [purif_ok_prob(model, input_fidelities[c[0]][1], input_fidelities[c[1]][1]) for c in choices]
    choices_res_fidelities: list[tuple[str, float]] = [(
            encode_purified_pair(input_fidelities[c[0]][0],input_fidelities[c[1]][0]),
            purif_res_fidelity(model, input_fidelities[c[0]][1], input_fidelities[c[1]][1])
        ) for c in choices]
    
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
        outcome_fidelities: list[tuple[str, float]] = input_fidelities.copy()
        new_fidelities: list[tuple[str, float]] = []
        for choice_i in range(len(choices)):
            c = choices[choice_i]
            choice_outcome = outcome_bitstring[choice_i]
            if choice_outcome is True:
                new_fidelities += [choices_res_fidelities[choice_i]]
            outcome_fidelities[c[0]] = (outcome_fidelities[c[0]][0], -1)
            outcome_fidelities[c[1]] = (outcome_fidelities[c[1]][0], -1)
        outcome_fidelities = [f for f in outcome_fidelities if f[1] >= 0] # filter out the -1s
        outcome_fidelities += new_fidelities

        outcome_fidelities = sort_str_named_list(outcome_fidelities)

        # Filter usable pairs based on the fidelity threshold
        outcome_usable_pairs, outcome_filtered_fidelities = filter_usable_pairs(outcome_fidelities, fidelity_threshold)

        list_after_current_step += [(outcome_probability, (outcome_usable_pairs, previous_iterations+1, outcome_filtered_fidelities))]


    list_after_recursion: list[tuple[float, tuple[int, int, list[tuple[str, float]]]]] = []
    for current_outcome_prob, (current_outcome_usable, current_outcome_iter, current_outcome_remaining_fids) in list_after_current_step:
        recursion_results = exact_recursive_simulation(policy, current_outcome_remaining_fids, fidelity_threshold, model, current_outcome_iter)
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

def average_usable_pairs_from_distribution(distribution: list[tuple[float, tuple[int, int, list[tuple[str, float]]]]]) -> float: 
    ret = 0.0
    for entry in distribution:
        prob = entry[0]
        usable = entry[1][0]
        ret += prob * usable
    return ret

def average_steps_from_distribution(distribution: list[tuple[float, tuple[int, int, list[tuple[str, float]]]]]) -> float: 
    ret = 0.0
    for entry in distribution:
        prob = entry[0]
        usable = entry[1][1]
        ret += prob * usable
    return ret

if __name__ == "__main__":
    threshold = 0.98
    model = PurificationModel.BIT_FLIP
    input_fid_list = gen_initial_named_pairs()
    generate_lookup_dict(input_fid_list, threshold, model)
    for policy in [lookup_policy, single_pair_greedy_policy_highest, single_pair_greedy_policy_lowest, all_pairs_policy_opposite]:
        end_distribution = exact_recursive_simulation(policy, input_fid_list, threshold, model)
        print(f"{policy.__name__}: {average_usable_pairs_from_distribution(end_distribution)} ({average_steps_from_distribution(end_distribution)} steps)")