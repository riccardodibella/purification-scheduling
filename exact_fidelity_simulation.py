# pyright: strict
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import chain, combinations, permutations, product
import math
from typing import Callable
from math import log10, ceil
import numpy as np
from enum import Enum, auto
import time # pyright: ignore[reportUnusedImport]
from functools import lru_cache # pyright: ignore[reportUnusedImport]

"""
# pyright: basic
from line_profiler import profile # PYTHONHASHSEED=0 kernprof -l -v exact_fidelity_simulation.py
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

PolicyFunction = Callable[[list[tuple[str, float]], float], list[tuple[int, int]]]

StateDescription = str
ChoiceDescription = str
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

def bit_flip_highest_deltaF_single_choice_policy(l: list[tuple[str, float]], thresh: float) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=True)

    best_delta_f: float = -1
    best_first_index: int = -1
    best_second_index: int = -1
    # lllll: list[tuple[tuple[float, float], float]] = []
    for first_index in range(0, len(working_l)-1):
        for second_index in range(first_index+1, len(working_l)):
            f1: float = working_l[first_index][0][1]
            f2: float = working_l[second_index][0][1]
            max_f1_f2 = max(f1, f2)
            res_fid = bit_flip_channel_purif_res_fidelity(f1, f2)
            delta_f = res_fid - max_f1_f2
            # print((f1, f2), "->", delta_f)
            # lllll.append(((f1, f2), delta_f))
            if delta_f > best_delta_f:
                best_delta_f = delta_f
                best_first_index = first_index
                best_second_index = second_index
    # lllll = sorted(lllll, key = lambda x: x[1], reverse=True)
    # print("------------------------")
    # print(l)
    # for aaa in lllll:
    #     print(aaa)
    # print("------------------------")
    assert best_delta_f >= 0
    assert best_first_index >= 0
    assert best_second_index >= 0
    # print((best_first_index, best_second_index), (working_l[best_first_index][1], working_l[best_second_index][1]), (0, len(l)-1))
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

def gen_initial_pairs() -> list[float]:
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

    print("generate_possible_states start")
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

def generate_possible_actions(state_str: StateDescription) -> list[ChoiceDescription]:
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
    



def generate_lookup_dict(initial_fids: list[tuple[str, float]], threshold: float, model: PurificationModel):
    # generate_immediate_termination_lookup_dict(initial_fids, threshold, model)

    possible_states: list[StateDescription] = generate_possible_states(initial_fids, threshold, model)
    print("generate_possible_states ok")
    """
    if SMART_PRUNING:
        possible_states = [state_string for state_string in possible_states if state_is_reachable(state_string, initial_fids, threshold, model)] # This shouldn't be needed anymore, it is already done inside generate_possible_states
        print("states pruning ok")
    """

    
    working_dict: dict[StateDescription, WorkingDictEntry] = {}
    for state_string in possible_states:
        assert state_string not in working_dict # if we catch a duplicated state string, we need to add a de-duplication step (with a set) at the end of generate_possible_states
        only_action_stop = force_only_action_stop(initial_fids, threshold, model, state_string)
        
        if only_action_stop:
            actions = [""]
        else:
            actions: list[ChoiceDescription] = generate_possible_actions(state_string)
        working_dict[state_string] = WorkingDictEntry(action=None, definitive=False, possible_actions=actions)

    print("generate_possible_actions ok")

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
            config_count_guess = int(config_count_guess*np.random.uniform(0.9, 1.15))
            # config_count_guess += np.random.randint(1, max(min(np.iinfo(np.int16).max, config_count_guess), 3))
            config_count_guess += 3
        config_count = min(config_count_guess, config_count)

        for i in np.logspace(log10(1), log10(config_count), 1000):
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
    while config_i < config_count:
        if config_i % 1_000 == 0:
            # print(f"{config_i}/{config_count} ({config_i/config_count*100}%)")
            print(f"{config_i} (max {config_count})")

        valid = set_nth_policy_blind_mod(config_i, working_dict, possible_states)
        if not valid:
            print(f"Stopped search early at {config_i}")
            break

        end_distribution = exact_recursive_simulation(lookup_policy, initial_fids, threshold, model)
        avg_usable = average_usable_pairs_from_distribution(end_distribution)
        avg_steps = average_steps_from_distribution(end_distribution)
        if(avg_usable > best_config_i_usable or (avg_usable == best_config_i_usable and avg_steps < best_config_i_steps)):
            best_config_i = config_i
            best_config_i_usable = avg_usable
            best_config_i_steps = avg_steps
        
        config_i += 1
    
    print(f"Total configurations traversed: {config_i+1}")
    
    print(f"Best configuration index: {best_config_i}")
    valid = set_nth_policy_blind_mod(best_config_i, working_dict, possible_states)
    assert valid is True
    return

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

if __name__ == "__main__":
    prog_start_time = time.time()
    threshold = 0.925
    model = PurificationModel.BIT_FLIP
    input_fid_list = gen_initial_named_pairs()
    generate_lookup_dict(input_fid_list, threshold, model)
    for policy in [lookup_policy, single_pair_greedy_policy_highest, single_pair_greedy_policy_lowest, all_pairs_policy_opposite, bit_flip_highest_deltaF_single_choice_policy]:
        end_distribution = exact_recursive_simulation(policy, input_fid_list, threshold, model)
        print(f"{policy.__name__}: {average_usable_pairs_from_distribution(end_distribution)} ({average_steps_from_distribution(end_distribution)} steps)")
    prog_end_time = time.time()
    print(f"Total execution time: {prog_end_time - prog_start_time} s")