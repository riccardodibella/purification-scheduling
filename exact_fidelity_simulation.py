# pyright: strict
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import chain, combinations, permutations, product
import math
from typing import Callable
import numpy as np
from enum import Enum, auto
import time
from functools import lru_cache # pyright: ignore[reportUnusedImport]
import matplotlib.pyplot as plt

"""
# pyright: basic
from line_profiler import profile # PYTHONHASHSEED=0 PYTHONOPTIMIZE=1 kernprof -l -v exact_fidelity_simulation.py
"""

import os
import sys

sys.set_int_max_str_digits(1_000_000)

if os.environ.get("PYTHONHASHSEED") != "0":
    print("Restarting and setting hash seed")
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

rng = np.random.default_rng(0)

SMART_PRUNING: bool = True
RANDOMIZED_LOWER_CONFIG_COUNT: bool = True
BADPATCH_SAFETY_COEFF: int = 1000 # BADPATCH

ULP_UNITS_EQUALITY_TOLERANCE = 5

PolicyFunction = Callable[[list[tuple[str, float]], float], list[tuple[int, int]]]

StateDescription = str
ChoiceDescription = str

ActionsGenerator = Callable[[StateDescription], list[ChoiceDescription]]

lookup_dict: dict[StateDescription, ChoiceDescription] = {}

def sort_fid_named_list(l: list[tuple[str, float]], highestFirst: bool = True) -> list[tuple[str, float]]:
    return sorted(l, key=lambda x: x[1], reverse=highestFirst)

def sort_str_named_list(l: list[tuple[str, float]]) -> list[tuple[str, float]]:
    # Lexicographic ascending order
    return sorted(l, key=lambda x: x[0], reverse=False)

def encode_state_description_from_sorted_list_str(l: list[str]) -> StateDescription:
    return ','.join(l)

def encode_state_description(l: list[tuple[str, float]]) -> StateDescription:
    l = sort_str_named_list(l)
    return encode_state_description_from_sorted_list_str([t[0] for t in l])

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
        print("lookup_policy state not found error")
        print(input_state)
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

def gen_initial_pairs() -> list[float]:
    return [0.91, 0.88, 0.85, 0.8]
    return [0.85, 0.8, 0.72, 0.7, 0.6]
    # return [0.924, 0.923, 0.922, 0.922, 0.921, 0.92, 0.919, 0.918]
    # return [0.92, 0.915, 0.91, 0.905, 0.9025, 0.90, 0.895, 0.89]
    return [0.92, 0.915, 0.91, 0.905, 0.90, 0.895, 0.89]
    # return [0.9, 0.88, 0.85, 0.8, 0.7, 0.6, 0.51, 0.5]
    # return [0.88, 0.85, 0.8, 0.7, 0.6, 0.55]
    # return [0.88, 0.85, 0.8, 0.7, 0.6]
    # return [0.88, 0.85, 0.8, 0.7]
    # return [0.88, 0.85, 0.8]
    # return [0.9, 0.9]

def gen_initial_named_pairs(pair_generator: Callable[[], list[float]] = gen_initial_pairs) -> list[tuple[str, float]]:
    fids: list[float] = pair_generator()
    fids = sorted(fids, reverse=True)
    num_chars = math.ceil(math.log10(len(fids)))
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

def bit_flip_highest_deltaF_single_choice_policy(l: list[tuple[str, float]], thresh: float) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=True)

    best_delta_f: float = -1
    best_first_index: int = -1
    best_second_index: int = -1
    for first_index in range(0, len(working_l)-1):
        for second_index in range(first_index+1, len(working_l)):
            f1: float = working_l[first_index][0][1]
            f2: float = working_l[second_index][0][1]
            max_f1_f2 = max(f1, f2)
            res_fid = bit_flip_channel_purif_res_fidelity(f1, f2)
            delta_f = res_fid - max_f1_f2
            if delta_f > best_delta_f:
                best_delta_f = delta_f
                best_first_index = first_index
                best_second_index = second_index
    assert best_delta_f >= 0
    assert best_first_index >= 0
    assert best_second_index >= 0
    return [(working_l[best_first_index][1], working_l[best_second_index][1])]

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

from typing import Any # pyright: ignore[reportUnusedImport]
# Tree = Any
type Tree = str | tuple[Tree, Tree]

# Note: with the current implementation, if the return boolean value is True the fidelity value is meaningless, for optimization reasons
def is_tree_or_subtree_above_threshold(tree: Tree, initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel) -> tuple[bool, float]:
    if type(tree) == str:
        if "+" not in tree:
            fid: None | float = None
            for key, f in initial_fids:
                if key == tree:
                    fid = f
                    break
            assert fid is not None
            return False, fid # individual inputs can never be above the threshold (common assumption in the code)
        else:
            fid = get_key_fidelity_recursive(tree, initial_fids, model)
            return fid >= threshold, fid
    
    assert type(tree) == tuple
    left_above, left_fid = is_tree_or_subtree_above_threshold(tree[0], initial_fids, threshold, model)
    if left_above:
        return True, 0.5
    right_above, right_fid = is_tree_or_subtree_above_threshold(tree[1], initial_fids, threshold, model)
    if right_above:
        return True, 0.5
    new_fid = purif_res_fidelity(model, left_fid, right_fid)
    return new_fid >= threshold, new_fid


@lru_cache(maxsize=None)
def collapse_tree_to_string(t: Tree) -> str:
    l_side: str = t[0] if type(t[0]) is str else collapse_tree_to_string(t[0])
    r_side: str = t[1] if type(t[1]) is str else collapse_tree_to_string(t[1])
    return encode_purified_pair(l_side, r_side)

@lru_cache(maxsize=None)
def all_trees(elements: tuple[str, ...]) -> list[Tree]:
    # https://claude.ai/share/3306902d-8459-40ca-b9d6-5f2770203f55
    if len(elements) == 1:
        return [elements[0]]

    result: list[Tree] = []
    for i in range(1, len(elements)):
        left_trees  = all_trees(elements[:i])
        right_trees = all_trees(elements[i:])
        for left in left_trees:
            for right in right_trees:
                result.append((left, right))
    return result

def generate_possible_states(initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel) -> list[StateDescription]:
    to_return: list[str] = []

    inputs: list[str] = [f[0] for f in initial_fids]

    # 1: Generate all possible pairs that we could arrive at
    all_possible_single_pair_strings: set[str] = set()
    possible_pair_subsets = str_powerset(inputs)
    for subset in possible_pair_subsets:

        def possible_orderings(input: tuple[str, ...]) -> list[tuple[str, ...]]:
            # https://claude.ai/share/669ba829-d830-4f13-a299-101e3a7c1a67
            return list(permutations(input))
        
        p_orderings = possible_orderings(subset)
        for ordering in p_orderings:
            
            a_trees =  all_trees(ordering)
            for tree in a_trees:
                # if "tree" is just a string, it is a pair by itself: add it to the set
                if type(tree) is str:
                    all_possible_single_pair_strings.add(tree)
                else:
                    # we have more than 1 pair in this combination: calculate the resulting state string and add it to the set
                    assert type(tree) is tuple
                    if is_tree_or_subtree_above_threshold(tree, initial_fids, threshold, model)[0]: # This is not a smart pruning, those states are actually unreachable
                        continue
                    else:
                        resulting_string = collapse_tree_to_string(tree)
                        all_possible_single_pair_strings.add(resulting_string)


    # 2: Construct all the possible lists of pairs without reusing the same input elements
    # https://claude.ai/share/b1c541f8-d1f5-4026-a939-9334c7488802
    input_uses_groupings: defaultdict[frozenset[str], list[str]] = defaultdict(list) # we use a defaultdict to make the "append" operation easier without checks for key present or absent
    for elem in all_possible_single_pair_strings:
        plus_delimited: str = elem.replace("<", "+").replace(">", "+")
        elem_inputs = plus_delimited.split("+")
        input_elems_set: set[str] = set()
        for input_elem in elem_inputs:
            if input_elem != "":
                input_elems_set.add(input_elem)
        frozen_input_elems_set: frozenset[str] = frozenset(input_elems_set) # we use a frozenset as a key because it is just an immutable set
        input_uses_groupings[frozen_input_elems_set].append(elem)
    
    frozenset_keys = list(input_uses_groupings.keys())
    
    # Remove the states that can are missing just one input: for each failed purification, we will lose at least 2 inputs
    frozenset_keys = [fs for fs in frozenset_keys if len(fs) != len(inputs) - 1]

    # print("Slow part start")
    # _start = time.time()

    """
    valid_key_combinations: list[tuple[frozenset[str], ...]] = []

    def str_frozenset_powerset(iterable: list[frozenset[str]]) -> chain[tuple[frozenset[str], ...]]:
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    all_key_combinations = str_frozenset_powerset(iterable=frozenset_keys)
    valid_key_combinations: list[tuple[frozenset[str], ...]] = []
    for key_combination in all_key_combinations:
        if len(key_combination) == 0:
            continue
        input_overlap = False
        found_keys_set: set[str] = set()
        for fs in key_combination:
            if not found_keys_set.isdisjoint(fs):
                input_overlap = True
                break
            found_keys_set |= fs
        if not input_overlap:
            valid_key_combinations.append(key_combination)
    """

    def recursive_build_valid_key_combinations(all_frozensets: list[frozenset[str]], start_index: int, current: list[frozenset[str]], to_return: list[tuple[frozenset[str], ...]]) -> list[tuple[frozenset[str], ...]]:
        used_keys:  frozenset[str] = frozenset()
        for c in current:
            used_keys = used_keys | c
        for index in range(start_index, len(all_frozensets)):
            fs_at_index = all_frozensets[index]
            if used_keys.isdisjoint(fs_at_index):
                current.append(fs_at_index)
                to_return.append(tuple(current))
                recursive_build_valid_key_combinations(all_frozensets, index+1, current, to_return)
                current.pop()
        return to_return

    valid_key_combinations: list[tuple[frozenset[str], ...]] = recursive_build_valid_key_combinations(all_frozensets = frozenset_keys, start_index = 0, current = [], to_return = [])

    # _end = time.time()
    # print(f"Slow part end: {_end - _start} s")

    all_valid_combination_lists: list[tuple[str, ...]] = []
    for comb in valid_key_combinations:
        working_list: list[tuple[str, ...]] = []
        for fs in comb:
            if len(working_list) == 0:
                for single_string in input_uses_groupings[fs]:
                    working_list.append((single_string,))
            else:
                new_working_list: list[tuple[str, ...]] = []
                for a in working_list:
                    for b in input_uses_groupings[fs]:
                        new_working_list.append(a + (b,))
                working_list = new_working_list
        all_valid_combination_lists += working_list

    # 3: Sort the elements of each list lexicographically
    lex_sorted_combination_lists = [sorted([*combination_tuple], reverse=False) for combination_tuple in all_valid_combination_lists]
    for i in range(len(all_valid_combination_lists)):
        all_valid_combination_lists[i] = tuple(sorted([*all_valid_combination_lists[i]], reverse=False)) # Lexicographic ascending order
    
    # 4: Merge each list in a single string and append it
    for sorted_combination in lex_sorted_combination_lists:
        to_return.append(encode_state_description_from_sorted_list_str(sorted_combination))
    return to_return

def generate_all_possible_actions(state_str: StateDescription) -> list[ChoiceDescription]:
    input_states: list[str] = state_str.split(",")
    if len(input_states) < 2:
        return [""]
    to_return: list[str] = []
    all_possible_single_pairs: list[tuple[int, int]] = list(combinations(range(len(input_states)), 2))
    def tuple_int_int_powerset(l: list[tuple[int, int]]) -> chain[tuple[tuple[int, int], ...]]:
        return chain.from_iterable(combinations(l, r) for r in range(len(l)+1))
    single_pairs_powerset = tuple_int_int_powerset(all_possible_single_pairs)
    for pairs_list in single_pairs_powerset:
        seen_set: set[int] = set()
        overlapping: bool = False
        for pair in pairs_list:
            if pair[0] in seen_set or pair[1] in seen_set:
                overlapping = True
                break
            seen_set.add(pair[0])
            seen_set.add(pair[1])
        
        if not overlapping:
            new_pairs_list: list[tuple[str, str]] = []
            for pair in pairs_list:
                new_pairs_list.append((input_states[pair[0]], input_states[pair[1]]))
            
            choice_string: ChoiceDescription = ""
            for i, p in enumerate(new_pairs_list):
                choice_string += f"{p[0]}:{p[1]}"
                if i < len(new_pairs_list) - 1:
                    choice_string += ","
            to_return.append(choice_string)
    return to_return

def generate_single_pair_actions(state_str: StateDescription) -> list[ChoiceDescription]:
    input_states: list[str] = state_str.split(",")
    if len(input_states) < 2:
        return [""]
    to_return: list[ChoiceDescription] = [""] # always include the "stop now" action
    all_possible_single_pairs: list[tuple[int, int]] = list(combinations(range(len(input_states)), 2))
    for index_a, index_b in all_possible_single_pairs:
        to_return.append(f"{input_states[index_a]}:{input_states[index_b]}")
    return to_return

def get_sorted_fid_generator(initial_fids: list[tuple[str, float]], model: PurificationModel):
    tuple_initial_fids: tuple[tuple[str, float], ...] = tuple(initial_fids)
    def sorted_fid_generator(state_str: StateDescription) -> list[ChoiceDescription]:
        input_states: list[str] = state_str.split(",")
        if len(input_states) < 2:
            return [""]
        states_with_fid: list[tuple[str, float]] = [(key, get_key_fidelity_recursive_tuple_fids(key, tuple_initial_fids, model)) for key in input_states]
        states_with_fid = sort_fid_named_list(states_with_fid, highestFirst=True)
        to_return: list[ChoiceDescription] = [""]

        working_string = ""
        for how_many_to_take in range(0, len(states_with_fid) // 2):
            if working_string != "":
                working_string += ","
            a: str = states_with_fid[2*how_many_to_take][0]
            b: str = states_with_fid[2*how_many_to_take+1][0]
            working_string += f"{a}:{b}"
                
            to_return.append(working_string)
        
        # print(f"SFG {state_str} {to_return}")
        return to_return
    return sorted_fid_generator

def get_sorted_increment_generator(initial_fids: list[tuple[str, float]], model: PurificationModel):
    tuple_initial_fids: tuple[tuple[str, float], ...] = tuple(initial_fids)
    def sorted_increment_generator(state_str: StateDescription) -> list[ChoiceDescription]:
        input_states: list[str] = state_str.split(",")
        if len(input_states) < 2:
            return [""]
        states_with_fid: list[tuple[str, float]] = [(key, get_key_fidelity_recursive_tuple_fids(key, tuple_initial_fids, model)) for key in input_states]

        to_return: list[ChoiceDescription] = [""]
        
        working_string = ""
        while len(states_with_fid) >= 2:
            all_possible_single_pairs: list[tuple[int, int]] = list(combinations(range(len(states_with_fid)), 2))
            chosen_pair: None | tuple[int, int] = None
            best_increment: float = -math.inf
            for pair in all_possible_single_pairs:
                fid_a: float = states_with_fid[pair[0]][1]
                fid_b: float = states_with_fid[pair[1]][1]
                max_fid: float = max(fid_a, fid_b)
                out_fid: float = purif_res_fidelity(model, fid_a, fid_b)
                increment = out_fid - max_fid
                if increment > best_increment:
                    chosen_pair = pair
                    best_increment = increment
            assert chosen_pair is not None
            if best_increment < 0: # This could happen under the werner model
                break

            if working_string != "":
                working_string += ","
            
            a: str = states_with_fid[chosen_pair[0]][0]
            b: str = states_with_fid[chosen_pair[1]][0]
            working_string += f"{a}:{b}"
            to_return.append(working_string)

            # check that the second index is higher, so if we remove it the first one is still valid for removal; 
            # this should be true because of how we build all_possible_single_pairs
            assert chosen_pair[1] > chosen_pair[0]
            # https://stackoverflow.com/a/11303234
            del states_with_fid[chosen_pair[1]]
            del states_with_fid[chosen_pair[0]]
        # print(f"SIG {state_str} {to_return}")
        return to_return
    return sorted_increment_generator

def remove_reduntant_actions(input: list[ChoiceDescription]) -> list[ChoiceDescription]:
    the_set: set[frozenset[str]] = set()
    for c in input:
        the_set.add(frozenset(c.split(",")))
    to_return: list[ChoiceDescription] = []
    for fs in the_set:
        choices: list[str] = sorted(list(fs))
        working = ""
        for k in choices:
            if working != "":
                working += ","
            working += k
        to_return.append(working)
    to_return = sorted(to_return, key=lambda s: len(s))
    return to_return

def get_sorted_fid_increment_generator(initial_fids: list[tuple[str, float]], model: PurificationModel):
    tuple_initial_fids: tuple[tuple[str, float], ...] = tuple(initial_fids)

    def _f(states_with_fid: list[tuple[str, float]], starting_string: str) -> list[ChoiceDescription]:
        if len(states_with_fid) < 2:
            return []
        if starting_string != "":
            starting_string += ","
                
        fid_states_copy: list[tuple[str, float]] = states_with_fid.copy()
        inc_states_copy: list[tuple[str, float]] = states_with_fid.copy()

        # Working pair with the highest fidelity
        fid_states_copy = sort_fid_named_list(fid_states_copy, highestFirst=True)
        fid_working_str = starting_string + f"{fid_states_copy[0][0]}:{fid_states_copy[1][0]}"
        fid_states_copy = fid_states_copy[2:]

        to_return: list[ChoiceDescription] = []
        to_return.append(fid_working_str)
        fid_additional_actions = _f(fid_states_copy, fid_working_str)
        inc_additional_actions: list[ChoiceDescription] = []


        all_possible_single_pairs: list[tuple[int, int]] = list(combinations(range(len(inc_states_copy)), 2))
        chosen_pair: None | tuple[int, int] = None
        best_increment: float = -math.inf
        for pair in all_possible_single_pairs:
            fid_a: float = inc_states_copy[pair[0]][1]
            fid_b: float = inc_states_copy[pair[1]][1]
            max_fid: float = max(fid_a, fid_b)
            out_fid: float = purif_res_fidelity(model, fid_a, fid_b)
            increment = out_fid - max_fid
            if increment > best_increment:
                chosen_pair = pair
                best_increment = increment
        assert chosen_pair is not None
        if best_increment >= 0: # This may not happen under the werner model
            inc_working_str = starting_string + f"{inc_states_copy[chosen_pair[0]][0]}:{inc_states_copy[chosen_pair[1]][0]}"
            assert chosen_pair[1] > chosen_pair[0]
            del inc_states_copy[chosen_pair[1]]
            del inc_states_copy[chosen_pair[0]]
            to_return.append(inc_working_str)
            inc_additional_actions = _f(inc_states_copy, inc_working_str)
        
        to_return += fid_additional_actions
        to_return += inc_additional_actions

        return to_return
    
    def sorted_fid_increment_generator(state_str: StateDescription) -> list[ChoiceDescription]:
        input_states: list[str] = state_str.split(",")
        if len(input_states) < 2:
            return [""]
        states_with_fid: list[tuple[str, float]] = [(key, get_key_fidelity_recursive_tuple_fids(key, tuple_initial_fids, model)) for key in input_states]

        to_return: list[ChoiceDescription] = [""]
        generated: list[ChoiceDescription] = _f(states_with_fid, "")
        to_return += generated

        to_return = remove_reduntant_actions(to_return)

        # print(f"SFI {state_str} {to_return}")
        return to_return
    return sorted_fid_increment_generator

@lru_cache(maxsize=None)
def get_key_fidelity_recursive_tuple_fids(key: str, initial_fids: tuple[tuple[str, float], ...], model: PurificationModel) -> float:
    assert key != ""
    if key[0] != "<":
        # Base case: search it directly in the array and return its fidelity
        for key2, fid in initial_fids:
            if key == key2:
                return fid
        # We didn't find the key... This is a problem.
        assert False
    
    # Remove first "<" and last ">"
    assert len(key) >= 5 # at least <X+X>
    assert key[0] == "<"
    key = key[1:]
    assert key[-1] == ">"
    key = key[:-1]

    # Split in the middle
    left_end = 0
    height = 1 if key[left_end] == "<" else 0
    while left_end == 0 or height > 0 or key[left_end] != "+":
        left_end += 1
        if key[left_end] == "<":
            height += 1
        elif key[left_end] == ">":
            height -= 1

    left_key = key[:left_end]
    right_key = key[left_end+1:]

    left_fid = get_key_fidelity_recursive_tuple_fids(left_key, initial_fids, model)
    right_fid = get_key_fidelity_recursive_tuple_fids(right_key, initial_fids, model)

    return purif_res_fidelity(model, left_fid, right_fid)

def get_key_fidelity_recursive(key: str, initial_fids: list[tuple[str, float]], model: PurificationModel) -> float:
    return get_key_fidelity_recursive_tuple_fids(key, tuple(initial_fids), model)


def is_state_above_threshold(key: str, initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel) -> bool:
    return get_key_fidelity_recursive(key, initial_fids, model) >= threshold

def state_is_reachable(state_string: StateDescription, initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel) -> bool:
    # we return False if some state in the state string has fidelity > threshold
    inputs: list[str] = state_string.split(",")
    # print(inputs)
    for input in inputs:
        if is_state_above_threshold(input, initial_fids, threshold, model):
            return False
    return True





def set_stop_policy_to_all(working_dict: dict[StateDescription, WorkingDictEntry]):
    """
    This acts as a safety net in case we stop early in the lookup_dict construction (because the residual counter reaches 0).
    In this case, all the keys that were not traversed will have their initial value, which is always the instant termination.
    """
    for k in working_dict.keys():

        # Safety check that we are actually setting the first choice (which should be "") as the default
        available_choices = working_dict[k].possible_actions
        assert available_choices is not None
        assert len(available_choices) > 0
        assert available_choices[0] == ""

        lookup_dict[k] = ""

def set_nth_policy_blind(target_config_number: int, working_dict: dict[StateDescription, WorkingDictEntry], possible_states: list[StateDescription]) -> None:
    residual_counter = target_config_number
    for state_string in possible_states:
        actions_for_this_state = working_dict[state_string].possible_actions
        assert actions_for_this_state is not None
        num_actions = len(actions_for_this_state)
        current_choice_index = residual_counter % num_actions
        lookup_dict[state_string] = actions_for_this_state[current_choice_index]
        residual_counter //= num_actions
        assert residual_counter >= 0

def set_nth_policy_blind_mod(target_config_number: int, working_dict: dict[StateDescription, WorkingDictEntry], possible_states: list[StateDescription]) -> bool:
    working_possible_states: deque[StateDescription] = deque(sorted(possible_states, key=lambda str: str.count(","), reverse=True))
    set_stop_policy_to_all(working_dict)

    residual_counter = target_config_number
    iter_num = 0
    while len(working_possible_states) > 0:
        state_string = working_possible_states.popleft()
        actions_for_this_state = working_dict[state_string].possible_actions
        assert actions_for_this_state is not None
        num_actions = len(actions_for_this_state)
        assert num_actions > 0
        current_choice_index = residual_counter % num_actions
        chosen_action: ChoiceDescription = actions_for_this_state[current_choice_index]
        lookup_dict[state_string] = chosen_action
        residual_counter //= num_actions
        assert residual_counter >= 0

        if iter_num == 0 and chosen_action != "":
            components = chosen_action.split(":")
            
            working_possible_states = deque(
                [s for s in working_possible_states if 
                    ((components[0]+"," not in s) and (","+components[0] not in s))
                    and
                    ((components[1]+"," not in s) and (","+components[1] not in s))
                    and
                    ((components[0] in s) == (components[1] in s))
                    and
                    (not (
                        (components[0] in s) and (components[1] in s) and (encode_purified_pair(components[0], components[1]) not in s)
                        ))
                ])

        if SMART_PRUNING:
            if chosen_action == "":
                inputs_for_this_state = set(state_string.split(","))
                new_working_deque: deque[StateDescription] = deque()
                for state_under_consideration in working_possible_states:
                    inputs_for_state_under_consideration = set(state_under_consideration.split(","))
                    proper_subset = True
                    for i in inputs_for_state_under_consideration:
                        if i not in inputs_for_this_state:
                            proper_subset = False
                            break
                    if not proper_subset:
                        new_working_deque.append(state_under_consideration)
                    else:
                        pass
                working_possible_states = new_working_deque
                

        iter_num+=1

        if residual_counter == 0:
            break

    if residual_counter > 0 and len(working_possible_states) == 0:
        return False
    return True

def all_purification_sequence_trees(inputs: list[str]) -> list[Tree]:
    # https://claude.ai/share/d3eb6410-3c94-4998-b610-59cf306537b4

    if len(inputs) == 1:
        return [inputs[0]]

    results: list[Tree] = []
    n = len(inputs)

    for left_size in range(1, n):
        for other_left_elements_positions in combinations(range(1, n), left_size - 1):
            left  = [inputs[0]] + [inputs[i] for i in other_left_elements_positions]
            right = [inputs[i] for i in range(1, n) if i not in other_left_elements_positions]

            left_trees = all_purification_sequence_trees(left)
            right_trees = all_purification_sequence_trees(right)
            for left_tree in left_trees:
                for right_tree in right_trees:
                    results.append((left_tree, right_tree))
    return results

def force_only_action_stop(initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel, state_string: str):
    if not SMART_PRUNING:
        return False

    # From here we do smart pruning
    inputs: list[str] = state_string.split(",")
    if(len(inputs) == 1):
        return True

    possible_trees: list[Tree] = all_purification_sequence_trees(inputs)
    keep: bool = False
    for t in possible_trees:
        assert type(t) is not str
        if is_tree_or_subtree_above_threshold(t, initial_fids, threshold, model)[0]:
            keep = True
            break
    return not keep
    



def generate_lookup_dict_BADPATCH(initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel):
    # generate_immediate_termination_lookup_dict(initial_fids, threshold, model)

    possible_states: list[StateDescription] = generate_possible_states(initial_fids, threshold, model)
    # print("generate_possible_states ok")

    
    working_dict: dict[StateDescription, WorkingDictEntry] = {}
    for state_string in possible_states:
        assert state_string not in working_dict # if we catch a duplicated state string, we need to add a de-duplication step (with a set) at the end of generate_possible_states
        only_action_stop = force_only_action_stop(initial_fids, threshold, model, state_string)
        
        if only_action_stop:
            actions = [""]
        else:
            actions: list[ChoiceDescription] = generate_all_possible_actions(state_string)
        working_dict[state_string] = WorkingDictEntry(action=None, definitive=False, possible_actions=actions)

    # print("generate_all_possible_actions ok")

    config_count = 1 # It is (should be...) a valid upper bound even for tree generation
    for state_string in possible_states:
        p_a = working_dict[state_string].possible_actions
        assert p_a is not None
        config_count *= len(p_a)

    entry_point = encode_state_description(initial_fids) # pyright: ignore[reportUnusedVariable]


    if RANDOMIZED_LOWER_CONFIG_COUNT:
        config_count_guess: int = 1
        while config_count_guess < config_count:
            valid = set_nth_policy_blind_mod(config_count_guess, working_dict, possible_states)
            if not valid:
                break
            config_count_guess = int(config_count_guess*rng.uniform(0.9, 1.15))
            # config_count_guess += rng.randint(1, max(min(np.iinfo(np.int16).max, config_count_guess), 3))
            config_count_guess += 3
        config_count = min(config_count_guess, config_count)

        for i in np.logspace(math.log10(1), math.log10(config_count), 1000):
            if i < 0:
                continue
            i = int(i)
            valid = set_nth_policy_blind_mod(i, working_dict, possible_states)
            if not valid:
                config_count = i
                break
    

    best_config_i: int = -1
    best_config_i_usable: float = -1.0
    best_config_i_steps: float = math.inf
    config_i: int = 0
    # print(f"BADPATCH {config_count} -> {config_count*BADPATCH_SAFETY_COEFF}") # BADPATCH
    config_count *= BADPATCH_SAFETY_COEFF # BADPATCH
    while config_i < config_count:
        if config_i % 10_000 == 0:
            # print(f"{config_i}/{config_count} ({config_i/config_count*100}%)")
            # print(f"{config_i} (max {config_count})")
            pass

        valid = set_nth_policy_blind_mod(config_i, working_dict, possible_states)
        if not valid:
            # print(f"Stopped search early at {config_i}")
            # break
            config_i += 1 # BADPATCH
            continue # BADPATCH

        end_distribution = exact_recursive_simulation(lookup_policy, initial_fids, threshold, model)
        avg_usable = average_usable_pairs_from_distribution(end_distribution)
        avg_steps = average_steps_from_distribution(end_distribution)
        if(avg_usable > best_config_i_usable or (avg_usable == best_config_i_usable and avg_steps < best_config_i_steps)):
            best_config_i = config_i
            best_config_i_usable = avg_usable
            best_config_i_steps = avg_steps
        
        config_i += 1
    
    # print(f"Total configurations traversed: {config_i+1}")
    
    # print(f"Best configuration index: {best_config_i}")
    valid = set_nth_policy_blind_mod(best_config_i, working_dict, possible_states)
    assert valid is True
    return


def generate_lookup_dict(initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel):

    possible_states: list[StateDescription] = generate_possible_states(initial_fids, threshold, model)
    # print("generate_possible_states ok")

    
    working_dict: dict[StateDescription, WorkingDictEntry] = {}
    for state_string in possible_states:
        assert state_string not in working_dict # if we catch a duplicated state string, we need to add a de-duplication step (with a set) at the end of generate_possible_states
        only_action_stop = force_only_action_stop(initial_fids, threshold, model, state_string)
        
        if only_action_stop:
            actions = [""]
        else:
            actions: list[ChoiceDescription] = generate_all_possible_actions(state_string)
        working_dict[state_string] = WorkingDictEntry(action=None, definitive=False, possible_actions=actions)

    # print("generate_all_possible_actions ok")

    config_count = 1 # It is (should be...) a valid upper bound even for tree generation
    for state_string in possible_states:
        p_a = working_dict[state_string].possible_actions
        assert p_a is not None
        config_count *= len(p_a)

    entry_point = encode_state_description(initial_fids) # pyright: ignore[reportUnusedVariable]

    best_config_i: int = -1
    best_config_i_usable: float = -1.0
    best_config_i_steps: float = math.inf
    config_i: int = 0
    while config_i < config_count:
        if config_i % 10_000 == 0:
            # print(f"{config_i}/{config_count} ({config_i/config_count*100}%)")
            # print(f"{config_i} (max {config_count})")
            pass

        set_nth_policy_blind(config_i, working_dict, possible_states)


        end_distribution = exact_recursive_simulation(lookup_policy, initial_fids, threshold, model)
        avg_usable = average_usable_pairs_from_distribution(end_distribution)
        avg_steps = average_steps_from_distribution(end_distribution)
        if(avg_usable > best_config_i_usable or (avg_usable == best_config_i_usable and avg_steps < best_config_i_steps)):
            best_config_i = config_i
            best_config_i_usable = avg_usable
            best_config_i_steps = avg_steps
        
        config_i += 1
    
    # print(f"Total configurations traversed: {config_i+1}")
    
    # print(f"Best configuration index: {best_config_i}")
    set_nth_policy_blind(best_config_i, working_dict, possible_states)
    return

class ActionItem:
    choice: ChoiceDescription

    # The first element is the bitstring (list of bools) associated with the outcome for that children
    # The second element is the probability of having this outcome
    # The third element is the number of usable pairs generated in that transition
    # The fourth element is the child node
    resulting_children: list[ tuple[ list[bool], float, int , DAGNode ] ]
    def __init__(self, choice: ChoiceDescription, resulting_children: list[ tuple[ list[bool], float, int , DAGNode ] ]) -> None:
        self.choice = choice
        self.resulting_children = resulting_children

class DAGNode:
    # Topological info
    state_string: StateDescription # str
    actions: list[ActionItem]
    actions_generated: bool # used as a safety check to ensure that we visit each node only once when we build the DAG structure

    # Search info
    best_action_chosen: bool
    chosen_action_index: int
    best_action_avg_usable: float
    best_action_avg_steps: float

    def __init__(self, state_string: StateDescription) -> None:
        self.state_string = state_string
        self.actions = []
        self.actions_generated = False

        self.best_action_chosen = False
        self.chosen_action_index = -1
        self.best_action_avg_usable = 0.0
        self.best_action_avg_steps = 0.0

    def add_action(self, action_item: ActionItem) -> None:
        self.actions.append(action_item)

    def set_chosen_action(self, index_or_choice_descr: int | ChoiceDescription, avg_usable: float, avg_steps: float):
        if isinstance(index_or_choice_descr, ChoiceDescription):
            index: int = -1
            for i, a in enumerate(self.actions):
                if a.choice == index_or_choice_descr:
                    index = i
                    break
        else:
            assert isinstance(index_or_choice_descr, int)
            index = index_or_choice_descr
        assert index >= 0
        assert index < len(self.actions)
        self.chosen_action_index = index
        self.best_action_chosen = True
        assert avg_usable >= 0
        assert avg_steps >= 0
        assert avg_steps != math.inf
        self.best_action_avg_usable = avg_usable
        self.best_action_avg_steps = avg_steps


class PurificationDAG:
    initial_pairs: list[tuple[str, float]]
    entry_point_string: StateDescription
    threshold: float
    root: DAGNode
    model: PurificationModel
    nodes_dict: dict[StateDescription, DAGNode]

    def __init__(self, initial_pairs: list[tuple[str, float]], threshold: float, model: PurificationModel, actions_generator: ActionsGenerator | None = None) -> None:
        self.initial_pairs = initial_pairs
        self.entry_point_string = encode_state_description(initial_pairs)
        self.threshold = threshold
        self.model = model
        self.nodes_dict = {}

        self.root = self.add_node(node_state_string=self.entry_point_string) # bootstrap the construction process

        if actions_generator is not None:
            self.construct_DAG(actions_generator)

    def add_node(self, node_state_string: StateDescription) -> DAGNode:
        if node_state_string not in self.nodes_dict:
            node = DAGNode(node_state_string)
            self.nodes_dict[node_state_string] = node
        else:
            node = self.nodes_dict[node_state_string]
        assert node is not None
        return node

    def construct_DAG(self, actions_generator: ActionsGenerator) -> None:
        assert self.root is not None
        assert self.root.actions_generated is False

        initial_pairs_tuple: tuple[tuple[str, float], ...] = tuple(self.initial_pairs)

        to_expand: set[str] = set()
        to_expand.add(self.entry_point_string)
        while len(to_expand) != 0:
            current_state_string = to_expand.pop()
            assert current_state_string in self.nodes_dict
            current_node: DAGNode = self.nodes_dict[current_state_string]

            assert not current_node.actions_generated

            actions: list[ChoiceDescription] = actions_generator(current_state_string)
            current_state_keys_set: set[str] = set(current_state_string.split(","))
            for action_string in actions:
                decoded_key_pairs: list[tuple[str, str]] = decode_choice_description(action_string)
                assert all([ x in current_state_keys_set and y in current_state_keys_set for (x, y) in decoded_key_pairs])
                resulting_children: list[tuple[list[bool], float, int, DAGNode]] = []
                num_action_choices: int = len(decoded_key_pairs)
                outcome_bitstrings: list[list[bool]] = bitstrings(num_action_choices)
                for bstring in outcome_bitstrings:
                    if len(bstring) == 0: # This handles the case where action_string == "" ("stop immediately" action)
                        continue
                    assert len(bstring) == len(decoded_key_pairs)
                    set_to_modify: set[str] = current_state_keys_set.copy()
                    generated_usable_pairs: int = 0
                    outcome_probability: float = 1.0
                    for i in range(len(bstring)):
                        input_keys: tuple[str, str] = decoded_key_pairs[i]
                        assert input_keys[0] in set_to_modify
                        assert input_keys[1] in set_to_modify
                        set_to_modify.remove(input_keys[0])
                        set_to_modify.remove(input_keys[1])

                        success: bool = bstring[i]
                        input_fid_0 = get_key_fidelity_recursive_tuple_fids(input_keys[0], initial_pairs_tuple, self.model)
                        input_fid_1 = get_key_fidelity_recursive_tuple_fids(input_keys[1], initial_pairs_tuple, self.model)
                        success_probability: float = purif_ok_prob(self.model, input_fid_0, input_fid_1)
                        if success:
                            new_key: str = encode_purified_pair(input_keys[0], input_keys[1])
                            if is_state_above_threshold(key=new_key, initial_fids=self.initial_pairs, threshold=self.threshold, model=self.model):
                                generated_usable_pairs += 1
                            else:
                                set_to_modify.add(new_key)
                            outcome_probability *= success_probability
                        else:
                            outcome_probability *= (1.0 - success_probability)
                    outcome_result_keys: list[str] = sorted(list(set_to_modify), reverse=False) # Lexicographic ascending order
                    outcome_result_str: StateDescription = encode_state_description_from_sorted_list_str(outcome_result_keys)
                    new_node: DAGNode = self.add_node(node_state_string=outcome_result_str)
                    resulting_children.append((bstring, outcome_probability, generated_usable_pairs, new_node))
                    if not new_node.actions_generated:
                        to_expand.add(new_node.state_string)
                if action_string == "":
                    assert len(resulting_children) == 0
                ai = ActionItem(action_string, resulting_children)
                current_node.add_action(ai)

            current_node.actions_generated = True
        # print("construct_DAG finished")

def within_equality_tolerance(a: float, b: float) -> bool:
    ulp_unit: float = math.ulp(max(abs(a), abs(b)))
    return abs(a - b) <= ULP_UNITS_EQUALITY_TOLERANCE*ulp_unit

def recursive_optimal_setup_core(dag: PurificationDAG, node: DAGNode) -> tuple[float, float]: # (avg_usable, avg_steps)
    assert node.state_string in dag.nodes_dict
    assert node is dag.nodes_dict[node.state_string] # exact equality of memory address; they must be the same object in memory (just a sanity check for my mental model)
    assert node.actions_generated

    if node.best_action_chosen:
        return (node.best_action_avg_usable, node.best_action_avg_steps)

    assert len(node.actions) > 0, "recursive_optimal_setup_core unexpected node with empty actions list, there should always be at least an empty \"\" action"

    best_action_index: int = -1
    best_avg_usable: float = -1
    best_avg_steps: float = math.inf

    for action_index, action in enumerate(node.actions):
        if action.choice == "":
            avg_usable = 0.0
            avg_steps = 0.0
        else:
            assert len(action.resulting_children) > 0
            avg_usable = 0.0
            avg_steps = 0.0
            for _, outcome_probability, outcome_usable, child_node in action.resulting_children:
                child_avg_usable, child_avg_steps = recursive_optimal_setup_core(dag, child_node)
                avg_usable += (outcome_usable + child_avg_usable) * outcome_probability
                avg_steps +=  child_avg_steps * outcome_probability
            avg_steps += 1 # include the cost of the current operation (which is not "stop immediately"), regardless of the outcomes and their probabilities
        
        if ((avg_usable > best_avg_usable) and not ((avg_steps > best_avg_steps) and (within_equality_tolerance(avg_usable, best_avg_usable)))) or ((avg_usable == best_avg_usable) and (avg_steps < best_avg_steps)) or (avg_usable < best_avg_usable and avg_steps < best_avg_steps and within_equality_tolerance(avg_usable, best_avg_usable)):
            best_avg_usable = avg_usable
            best_avg_steps = avg_steps
            best_action_index = action_index
            # print(f"current candidate\tstate \"{action.state_string}\" choice \"{action.choice}\" (index {action_index}) usable {avg_usable} steps {avg_steps}")
        else:
            # print(f"discarded choice\tstate \"{action.state_string}\" choice \"{action.choice}\" (index {action_index}) usable {avg_usable} steps {avg_steps}")
            pass

    # print(f"CHOSEN BEST ACTION\tstate {node.actions[best_action_index].state_string} choice {node.actions[best_action_index].choice} (index {best_action_index}) usable {best_avg_usable} steps {best_avg_steps}")
    node.set_chosen_action(best_action_index, avg_usable=best_avg_usable, avg_steps=best_avg_steps)   
    return (node.best_action_avg_usable, node.best_action_avg_steps)

def recursive_optimal_setup_main(dag: PurificationDAG) -> None:
    recursive_optimal_setup_core(dag, dag.root)

class PurificationDAGPolicy:
    dag: PurificationDAG
    __name__ = "PurificationDAGPolicy"
    def __init__(self, dag: PurificationDAG) -> None:
        self.dag = dag
    def __call__(self, l: list[tuple[str, float]], thresh: float) -> list[tuple[int, int]]:
        input_state: StateDescription = encode_state_description(l)
        assert input_state in self.dag.nodes_dict.keys(), f"PurificationDAGPolicy state |{input_state}| not found"
        node: DAGNode = self.dag.nodes_dict[input_state]
        assert node.best_action_chosen
        action_index: int = node.chosen_action_index
        assert action_index >= 0
        assert action_index < len(node.actions)
        choice_str: ChoiceDescription = node.actions[action_index].choice
        to_return = decode_choice(l, choice_str)
        return to_return


def exact_recursive_simulation(policy: PolicyFunction, input_fidelities: list[tuple[str, float]], fidelity_threshold: float, model: PurificationModel, previous_iterations: int = 0) -> list[tuple[float, tuple[int, int, list[tuple[str, float]]]]]:
    """
    Return type: [(probability, (# of usable pairs, # of iterations, [(remaining_keys, remaining_fids)]))]
    """
    if(len(input_fidelities) < 2):
        return [(1, (0, previous_iterations, input_fidelities))]
    
    list_after_current_step: list[tuple[float, tuple[int, int, list[tuple[str, float]]]]] = []
    choices = policy(input_fidelities, fidelity_threshold)
    assert check_feasible_schedule(choices)

    if len(choices) == 0:
        # empty choice list means that the purification path ends here and leftover pairs stay unused
        return [(1, (0, previous_iterations, input_fidelities))]

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
        ret += prob * float(usable)
    return ret

def average_steps_from_distribution(distribution: list[tuple[float, tuple[int, int, list[tuple[str, float]]]]]) -> float: 
    ret = 0.0
    for entry in distribution:
        prob = entry[0]
        steps = entry[1][1]
        ret += prob * float(steps)
    return ret


def small_input_high_fid_equality_test() -> None:
    prog_start_time = time.time()
    threshold = 0.925
    model = PurificationModel.BIT_FLIP
    NUM_TESTS = 100
    print("SMALL INPUT HIGH FIDELITY EQUALITY TEST")
    for i in range(NUM_TESTS):
        print(f"TEST {i+1}/{NUM_TESTS}")
        def _input_generator() -> list[float]:
            to_return = sorted([rng.uniform(0.9, threshold) for _ in range(4)], reverse=True)
            return to_return
        input_fid_list = gen_initial_named_pairs(_input_generator)

        dag: PurificationDAG = PurificationDAG(input_fid_list, threshold, model, generate_all_possible_actions)
        recursive_optimal_setup_main(dag)
        dag_policy = PurificationDAGPolicy(dag)
        res_dag = exact_recursive_simulation(dag_policy, input_fid_list, threshold, model)
        assert np.allclose([average_usable_pairs_from_distribution(res_dag), average_steps_from_distribution(res_dag)], [dag.root.best_action_avg_usable,dag.root.best_action_avg_steps])

        generate_lookup_dict(input_fid_list, threshold, model)
        res_dict = exact_recursive_simulation(lookup_policy, input_fid_list, threshold, model)

        generate_lookup_dict_BADPATCH(input_fid_list, threshold, model)
        res_dict_BADPATCH = exact_recursive_simulation(lookup_policy, input_fid_list, threshold, model)

        if not (
            (average_usable_pairs_from_distribution(res_dag) == average_usable_pairs_from_distribution(res_dict) and average_usable_pairs_from_distribution(res_dict) == average_usable_pairs_from_distribution(res_dict_BADPATCH))
            and
            (average_steps_from_distribution(res_dag) == average_steps_from_distribution(res_dict) and average_steps_from_distribution(res_dict) == average_steps_from_distribution(res_dict_BADPATCH))
        ):
            print(f"recursive_optimal_setup_main:\t{average_usable_pairs_from_distribution(res_dag)} ({average_steps_from_distribution(res_dag)} steps)")
            print(f"generate_lookup_dict:\t\t{average_usable_pairs_from_distribution(res_dict)} ({average_steps_from_distribution(res_dict)} steps)")
            print(f"generate_lookup_dict_BADPATCH:\t{average_usable_pairs_from_distribution(res_dict_BADPATCH)} ({average_steps_from_distribution(res_dict_BADPATCH)} steps)")
            print(f"1st term equality: {average_usable_pairs_from_distribution(res_dag) == average_usable_pairs_from_distribution(res_dict) and average_usable_pairs_from_distribution(res_dict) == average_usable_pairs_from_distribution(res_dict_BADPATCH)}")
            print(f"2nd term equality: {average_steps_from_distribution(res_dag) == average_steps_from_distribution(res_dict) and average_steps_from_distribution(res_dict) == average_steps_from_distribution(res_dict_BADPATCH)}")
            print(f"DAG root: {[dag.root.best_action_avg_usable,dag.root.best_action_avg_steps]}")

            print(res_dag)
            print(res_dict)
            print("TEST FAILED")
            sys.exit(0)
            break

    prog_end_time = time.time()
    print(f"Total execution time: {prog_end_time - prog_start_time} s")

def fidelity_increment_sorting_test() -> None:
    prog_start_time = time.time()
    NUM_TESTS = 100
    print("FIDELITY/INCREMENT SORTING TEST")
    for i in range(NUM_TESTS):
        print(f"TEST {i+1}/{NUM_TESTS}")
        threshold = rng.uniform(0.7, 0.95)
        model: PurificationModel = rng.choice(np.array([PurificationModel.BIT_FLIP, PurificationModel.WERNER]))
        num_pairs: int = int(rng.integers(8, 19 if model == PurificationModel.BIT_FLIP else 14))
        def _input_generator() -> list[float]:
            to_return = sorted([rng.uniform(0.6, threshold) for _ in range(num_pairs)], reverse=True)
            return to_return
        input_fid_list = gen_initial_named_pairs(_input_generator)

        fid_gen = get_sorted_fid_generator(input_fid_list, model)
        inc_gen = get_sorted_increment_generator(input_fid_list, model)
        both_gen = get_sorted_fid_increment_generator(input_fid_list, model)

        dag: PurificationDAG = PurificationDAG(input_fid_list, threshold, model, fid_gen)
        recursive_optimal_setup_main(dag)
        dag_policy = PurificationDAGPolicy(dag)
        res_fid = exact_recursive_simulation(dag_policy, input_fid_list, threshold, model)
        assert np.allclose([average_usable_pairs_from_distribution(res_fid), average_steps_from_distribution(res_fid)], [dag.root.best_action_avg_usable,dag.root.best_action_avg_steps])

        dag: PurificationDAG = PurificationDAG(input_fid_list, threshold, model, inc_gen)
        recursive_optimal_setup_main(dag)
        dag_policy = PurificationDAGPolicy(dag)
        res_inc = exact_recursive_simulation(dag_policy, input_fid_list, threshold, model)
        assert np.allclose([average_usable_pairs_from_distribution(res_inc), average_steps_from_distribution(res_inc)], [dag.root.best_action_avg_usable,dag.root.best_action_avg_steps])

        dag: PurificationDAG = PurificationDAG(input_fid_list, threshold, model, both_gen)
        recursive_optimal_setup_main(dag)
        dag_policy = PurificationDAGPolicy(dag)
        res_both = exact_recursive_simulation(dag_policy, input_fid_list, threshold, model)
        assert np.allclose([average_usable_pairs_from_distribution(res_both), average_steps_from_distribution(res_both)], [dag.root.best_action_avg_usable,dag.root.best_action_avg_steps])

        def is_actually_better(better: tuple[float, float], worse: tuple[float, float]):
            if better[0] > worse[0]:
                return True
            if better[0] < worse[0] and not within_equality_tolerance(better[0], worse[0]):
                return False
            assert within_equality_tolerance(better[0], worse[0])
            if better[1] < worse[1]:
                return True
            if better[1] > worse[1] and not within_equality_tolerance(better[0], worse[0]):
                return False
            assert within_equality_tolerance(better[1], worse[1])
            return True
        if (
            (not is_actually_better(
                (average_usable_pairs_from_distribution(res_both), average_steps_from_distribution(res_both)), 
                (average_usable_pairs_from_distribution(res_fid), average_steps_from_distribution(res_fid))
            )) 
            or
            (not is_actually_better(
                (average_usable_pairs_from_distribution(res_both), average_steps_from_distribution(res_both)), 
                (average_usable_pairs_from_distribution(res_inc), average_steps_from_distribution(res_inc))
            ))
            ):
            print(f"Configuration: threshold {threshold} model {model.name} initial_pairs {input_fid_list}")
            print(f"{fid_gen.__name__}:\t{average_usable_pairs_from_distribution(res_fid)} ({average_steps_from_distribution(res_fid)} steps)")
            print(f"{inc_gen.__name__}:\t\t{average_usable_pairs_from_distribution(res_inc)} ({average_steps_from_distribution(res_inc)} steps)")
            print(f"{both_gen.__name__}:\t{average_usable_pairs_from_distribution(res_both)} ({average_steps_from_distribution(res_both)} steps)")
            print("TEST FAILED")
            sys.exit(0)
            break

    prog_end_time = time.time()
    print(f"Total execution time: {prog_end_time - prog_start_time} s")

def playground_main() -> None:
    prog_start_time = time.time()
    threshold = 0.9
    model = PurificationModel.BIT_FLIP

    def _input_generator() -> list[float]:
        to_return = sorted([rng.uniform(0.6, threshold) for _ in range(10)], reverse=True)
        return to_return
    input_fid_list = gen_initial_named_pairs(_input_generator)

    actions_generators: list[ActionsGenerator] = [
            generate_single_pair_actions,
            get_sorted_fid_generator(input_fid_list, model), 
            get_sorted_increment_generator(input_fid_list, model), 
            get_sorted_fid_increment_generator(input_fid_list, model)
        ]
    for a_g in actions_generators:
        dag: PurificationDAG = PurificationDAG(input_fid_list, threshold, model, a_g)
        recursive_optimal_setup_main(dag)
        policy = PurificationDAGPolicy(dag)
        res = exact_recursive_simulation(policy, input_fid_list, threshold, model)
        assert np.allclose([average_usable_pairs_from_distribution(res), average_steps_from_distribution(res)], [dag.root.best_action_avg_usable,dag.root.best_action_avg_steps])

        print(f"{a_g.__name__.ljust(30)}: {average_usable_pairs_from_distribution(res)} ({average_steps_from_distribution(res)} steps)")

    prog_end_time = time.time()
    print(f"Total execution time: {prog_end_time - prog_start_time} s")

def progressive_increase_main() -> None:
    prog_start_time = time.time()
    threshold = 0.9
    model = PurificationModel.BIT_FLIP
    NUM_SAMPLES = 200
    MAX_PAIRS = 16

    gen_names: list[str] = [
        "all_single_pair",
        "sorted_fid",
        "sorted_increment",
        "sorted_fid_increment",
    ]
    gen_max_test_pairs: list[int] = [
        6,
        MAX_PAIRS,
        MAX_PAIRS,
        11
    ]
    num_pairs_range = list(range(2, MAX_PAIRS + 1))


    results: list[                  # first index is generator index
        list[                       # second index is index inside num_pairs_range
            list[                   # third index is sample_i
                tuple[float, float] # fourth index is 0 for "usable", 1 for "steps"
                ]
            ]
        ] = [[] for _ in gen_names]
    assert len(results) == len(gen_names)

    for num_pairs_range_index, num_pairs in enumerate(num_pairs_range):
        print(f"{num_pairs} PAIRS")
        def _input_generator() -> list[float]:
            to_return = sorted([rng.uniform(0.6, threshold) for _ in range(num_pairs)], reverse=True)
            return to_return

        for single_generator_results_list in results:
            assert len(single_generator_results_list) == num_pairs_range_index
            single_generator_results_list.append([])
        
        for sample_i in range(NUM_SAMPLES):
            input_fid_list = gen_initial_named_pairs(_input_generator)
            actions_generators: list[ActionsGenerator] = [
                generate_single_pair_actions,
                get_sorted_fid_generator(input_fid_list, model), 
                get_sorted_increment_generator(input_fid_list, model), 
                get_sorted_fid_increment_generator(input_fid_list, model)
            ]
            for gen_i in range(len(actions_generators)):
                a_g: ActionsGenerator = actions_generators[gen_i]
                gen_name: str = gen_names[gen_i]
                gen_max_inputs: float = gen_max_test_pairs[gen_i]
                if num_pairs > gen_max_inputs:
                    continue
                
                dag: PurificationDAG = PurificationDAG(input_fid_list, threshold, model, a_g)
                recursive_optimal_setup_main(dag)
                policy = PurificationDAGPolicy(dag)
                res = exact_recursive_simulation(policy, input_fid_list, threshold, model)
                assert np.allclose([average_usable_pairs_from_distribution(res), average_steps_from_distribution(res)], [dag.root.best_action_avg_usable,dag.root.best_action_avg_steps])
                usable: float = average_usable_pairs_from_distribution(res)
                steps: float = average_steps_from_distribution(res)

                target_res_list = results[gen_i][num_pairs_range_index]
                assert len(target_res_list) == sample_i
                target_res_list.append((usable, steps))
                assert len(results[gen_i][num_pairs_range_index]) == sample_i+1
    
    prog_end_time = time.time()
    print(f"Total execution time: {prog_end_time - prog_start_time} s")

    # --- Plotting ---
    plt.figure()  # pyright: ignore[reportUnknownMemberType]

    for gen_i, gen_name in enumerate(gen_names):

        average_usable_list: list[float] = []
        average_steps_list: list[float] = []
        num_pairs_list: list[int] = [] # keep only the relevant elements from num_pairs_range
        
        single_generator_results_list = results[gen_i]
        for num_pairs_range_index, samples in enumerate(single_generator_results_list):
            if len(samples) > 0:
                num_pairs = num_pairs_range[num_pairs_range_index]
                samples_usable: list[float] = [t[0] for t in samples]
                samples_steps: list[float] = [t[1] for t in samples]
                assert len(samples_usable) > 0 and len(samples_steps) > 0
                assert len(samples_usable) == len(samples_steps)

                avg_usable = sum(samples_usable) / len(samples_usable)
                avg_steps = sum(samples_steps) / len(samples_steps)

                num_pairs_list.append(num_pairs)
                average_usable_list.append(avg_usable)
                average_steps_list.append(avg_steps)

                assert len(average_usable_list) == len(average_steps_list) and len(average_usable_list) == len(num_pairs_list)

        
        # Connecting line
        plt.plot(   # pyright: ignore[reportUnknownMemberType]
            num_pairs_list,
            average_usable_list,
            label=gen_name,
            linewidth=0.5,
        )

        # Individual markers with different sizes
        plt.scatter(   # pyright: ignore[reportUnknownMemberType]
            num_pairs_list,
            average_usable_list,
            s=[(size*2)**2 for size in average_steps_list], # Area proportional to the number of steps
            label="_nolegend_"
        )

    plt.xlabel("Number of usable pairs")   # pyright: ignore[reportUnknownMemberType]
    plt.ylabel("Average usable pairs")  # pyright: ignore[reportUnknownMemberType]
    plt.title("Average usable pairs vs. number of input pairs")  # pyright: ignore[reportUnknownMemberType]
    plt.legend()  # pyright: ignore[reportUnknownMemberType]
    plt.grid(True)  # pyright: ignore[reportUnknownMemberType]
    plt.show()  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    # small_input_high_fid_equality_test()
    # fidelity_increment_sorting_test()
    # playground_main()
    progressive_increase_main()