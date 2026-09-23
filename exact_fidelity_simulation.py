# pyright: strict
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import chain, combinations, product
import json
import math
from typing import Callable
import numpy as np
from enum import Enum, auto
import time
from functools import cache # pyright: ignore[reportUnusedImport]


"""
# pyright: basic
from line_profiler import profile # PYTHONHASHSEED=0 PYTHONOPTIMIZE=1 kernprof -l -v exact_fidelity_simulation.py
"""


import os
import sys

# sys.set_int_max_str_digits(1_000_000)

if os.environ.get("PYTHONHASHSEED") != "0":
    print("Restarting and setting hash seed")
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)


ULP_UNITS_EQUALITY_TOLERANCE = 5

class PurificationModel(Enum):
    BIT_FLIP = auto(),
    WERNER = auto()

PolicyFunction = Callable[[list[tuple[str, float]], float, PurificationModel], list[tuple[int, int]]]

StateDescription = str
ChoiceDescription = str

ActionsGenerator = Callable[[StateDescription], list[ChoiceDescription]]

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

def single_pair_greedy_policy_highest(l: list[tuple[str, float]], thresh: float, model: PurificationModel) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=True)
    return [(working_l[0][1],working_l[1][1])]


def single_pair_greedy_policy_lowest(l: list[tuple[str, float]], thresh: float, model: PurificationModel) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=False)
    return [(working_l[0][1],working_l[1][1])]

def all_pairs_policy_opposite_middle_hole(l: list[tuple[str, float]], thresh: float, model: PurificationModel) -> list[tuple[int, int]]:
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

def all_pairs_policy_opposite_tail_hole(l: list[tuple[str, float]], thresh: float, model: PurificationModel) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=True)

    if len(working_l) % 2 == 1:
        working_l = working_l[:-1]
    
    pairs: list[tuple[int, int]] = []
    for i in range(0, int(len(working_l)/2)):
        idx1 = working_l[i][1]
        idx2 = working_l[len(working_l)-1-i][1]
        pairs += [(idx1, idx2)]
    return pairs

def all_pairs_policy_opposite_head_hole(l: list[tuple[str, float]], thresh: float, model: PurificationModel) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0][1], reverse=True)

    if len(working_l) % 2 == 1:
        working_l = working_l[1:]
    
    pairs: list[tuple[int, int]] = []
    for i in range(0, int(len(working_l)/2)):
        idx1 = working_l[i][1]
        idx2 = working_l[len(working_l)-1-i][1]
        pairs += [(idx1, idx2)]
    return pairs

def gen_initial_named_pairs(pair_generator: Callable[[], list[float]]) -> list[tuple[str, float]]:
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

def bit_flip_highest_deltaF_single_choice_policy(l: list[tuple[str, float]], thresh: float, model: PurificationModel) -> list[tuple[int, int]]:
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

def generate_single_pair_actions_inertia(state_str: StateDescription) -> list[ChoiceDescription]:
    if not "+" in state_str:
        return generate_single_pair_actions(state_str)
    
    input_states: list[str] = state_str.split(",")
    if len(input_states) < 2:
        return [""]

    merged_inputs: list[str] = [i_s for i_s in input_states if "+" in i_s]
    assert len(merged_inputs) == 1

    first_key: str = merged_inputs[0]

    non_merged_inputs: list[str] = [i_s for i_s in input_states if i_s != first_key]

    # This does not include the "stop now" action! If we already have purified something it is because some usable pair was achievable, so it doesn't make sense to stop now
    to_return: list[ChoiceDescription] = [f"{first_key}:{n_m_i}" for n_m_i in non_merged_inputs]

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

def get_highest_fid_single_choice_generator(initial_fids: list[tuple[str, float]], model: PurificationModel):
    tuple_initial_fids: tuple[tuple[str, float], ...] = tuple(initial_fids)
    def highest_fid_single_choice_generator(state_str: StateDescription) -> list[ChoiceDescription]:
        input_states: list[str] = state_str.split(",")
        if len(input_states) < 2:
            return [""]
        states_with_fid: list[tuple[str, float]] = [(key, get_key_fidelity_recursive_tuple_fids(key, tuple_initial_fids, model)) for key in input_states]
        states_with_fid = sort_fid_named_list(states_with_fid, highestFirst=True)
        to_return: list[ChoiceDescription] = [""]
        for i in range(1, len(states_with_fid)):
            a: str = states_with_fid[0][0]
            b: str = states_with_fid[i][0]
            to_return.append(f"{a}:{b}")
        return to_return
    return highest_fid_single_choice_generator

def get_lowest_fid_single_choice_generator(initial_fids: list[tuple[str, float]], model: PurificationModel):
    tuple_initial_fids: tuple[tuple[str, float], ...] = tuple(initial_fids)
    def lowest_fid_single_choice_generator(state_str: StateDescription) -> list[ChoiceDescription]:
        input_states: list[str] = state_str.split(",")
        if len(input_states) < 2:
            return [""]
        states_with_fid: list[tuple[str, float]] = [(key, get_key_fidelity_recursive_tuple_fids(key, tuple_initial_fids, model)) for key in input_states]
        states_with_fid = sort_fid_named_list(states_with_fid, highestFirst=False)
        to_return: list[ChoiceDescription] = [""]
        for i in range(1, len(states_with_fid)):
            a: str = states_with_fid[0][0]
            b: str = states_with_fid[i][0]
            to_return.append(f"{a}:{b}")
        return to_return
    return lowest_fid_single_choice_generator

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

@cache
def get_key_fidelity_recursive_tuple_fids(key: str, initial_fids: tuple[tuple[str, float], ...], model: PurificationModel) -> float:
    assert key != ""

    # Check added to handle the case where initial_fids is not actually the set of initial pairs, but can contain combinations.
    # This simplifies the code for the optimistic search policy, which can be treated as a pure policy without needing to inject initial_fids in it, as it can use the current states directly to do its calculations.
    # If this loop becomes a bottleneck down the line we can modify that policy and remove this here. However, this may never be a problem, since the cache may catch the vast majority of the calls anyways.
    for key2, fid in initial_fids:
        if key == key2:
            return fid

    
    if key[0] != "<":
        # Base case: search it directly in the array and return its fidelity
        for key2, fid in initial_fids:
            if key == key2:
                return fid
        # We didn't find the key... This is a problem.
        print("Problemi! Problemi! Problemi per get_key_fidelity_recursive_tuple_fids")
        print(f"key: |{key}|")
        print(f"initial fids: {initial_fids}")
        assert False
        exit(0)
    
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


cached_functions = [get_key_fidelity_recursive_tuple_fids]



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

def extract_immediate_choices_from_tree(tree: Tree) -> list[tuple[str, str]]:
    if type(tree) == str:
        return []
    assert type(tree) == tuple
    to_return: list[tuple[str, str]] = []
    left = tree[0]
    right = tree[1]
    if type(left) == str and type(right) == str:
        to_return += [(left, right)]
    to_return += extract_immediate_choices_from_tree(left)
    to_return += extract_immediate_choices_from_tree(right)
    return to_return
    
ChooseTreeFunction = Callable[[list[tuple[str, float]], list[Tree], PurificationModel], Tree]

def choose_tree_highest_fid(param_inputs: list[tuple[str, float]], param_trees: list[Tree], model: PurificationModel) -> Tree:
    assert len(param_trees) > 0
    working_candidates: list[Tree] = param_trees.copy()
    working_candidates = list(reversed(working_candidates))
    working_candidates = sorted(working_candidates, key=lambda t: is_tree_or_subtree_above_threshold(tree=t, initial_fids=param_inputs, threshold=math.inf, model=model)[1], reverse=True)
    return working_candidates[0]

def choose_tree_lowest_fid(param_inputs: list[tuple[str, float]], param_trees: list[Tree], model: PurificationModel) -> Tree:
    assert len(param_trees) > 0
    working_candidates: list[Tree] = param_trees.copy()
    working_candidates = list(reversed(working_candidates))
    working_candidates = sorted(working_candidates, key=lambda t: is_tree_or_subtree_above_threshold(tree=t, initial_fids=param_inputs, threshold=math.inf, model=model)[1], reverse=True)
    return working_candidates[-1]

def choose_tree_most_choices_highest_fid(param_inputs: list[tuple[str, float]], param_trees: list[Tree], model: PurificationModel) -> Tree:
    assert len(param_trees) > 0
    working_candidates: list[Tree] = param_trees.copy()
    working_candidates = list(reversed(working_candidates))
    working_candidates = sorted(working_candidates, key=lambda t: is_tree_or_subtree_above_threshold(tree=t, initial_fids=param_inputs, threshold=math.inf, model=model)[1], reverse=True)
    working_candidates = sorted(working_candidates, key=lambda t: len(extract_immediate_choices_from_tree(t)), reverse=True)
    return working_candidates[0]

def choose_tree_most_choices_lowest_fid(param_inputs: list[tuple[str, float]], param_trees: list[Tree], model: PurificationModel) -> Tree:
    assert len(param_trees) > 0
    working_candidates: list[Tree] = param_trees.copy()
    # working_candidates = list(reversed(working_candidates))
    working_candidates = sorted(working_candidates, key=lambda t: is_tree_or_subtree_above_threshold(tree=t, initial_fids=param_inputs, threshold=math.inf, model=model)[1])
    working_candidates = sorted(working_candidates, key=lambda t: len(extract_immediate_choices_from_tree(t)), reverse=True)
    return working_candidates[0]


def get_optimistic_search_policy(choose_tree: ChooseTreeFunction) -> Callable[[list[tuple[str, float]], float, PurificationModel], list[tuple[int, int]]]:
    def optimistic_search_policy(l: list[tuple[str, float]], thresh: float, model: PurificationModel) -> list[tuple[int, int]]:
        if(len(l) < 2):
            return []
        working_l = zip(l, list(range(len(l))))
        working_l = sorted(working_l, key=lambda x: x[0][1], reverse=True) # sorted in descending order

        candidate_trees: list[Tree] = []

        for number_of_pairs_considered in range(2, len(working_l) + 1):
            # print(f"number_of_pairs_considered {number_of_pairs_considered}")
            for working_l_chosen_elements in combinations(working_l, number_of_pairs_considered):
                keys: list[str] = [elem[0][0] for elem in working_l_chosen_elements]
                trees_that_use_all_keys: list[Tree] = all_purification_sequence_trees(keys)
                for current_tree in trees_that_use_all_keys:
                    if is_tree_or_subtree_above_threshold(current_tree, l, threshold=thresh, model=model)[0]:
                        candidate_trees.append(current_tree)
                if len(candidate_trees) > 0:
                    break
        if len(candidate_trees) > 0:
            chosen_tree = choose_tree(l, candidate_trees, model)
            direct_choices: list[tuple[str, str]] = extract_immediate_choices_from_tree(chosen_tree)

            to_return: list[tuple[int, int]] = []
            for choice in direct_choices:
                index_0: float = -1
                index_1: float = -1

                for working_l_element in working_l:
                    if choice[0] == working_l_element[0][0]:
                        assert index_0 < 0
                        index_0 = working_l_element[1]
                    if choice[1] == working_l_element[0][0]:
                        assert index_1 < 0
                        index_1 = working_l_element[1]
                assert index_0 >= 0 and index_1 >= 0

                to_return.append((index_0, index_1))

            return to_return
        return [] # It is impossible to arrive at a usable pair from here, so it is better to stop now
    return optimistic_search_policy



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
    def __call__(self, l: list[tuple[str, float]], thresh: float, model: PurificationModel) -> list[tuple[int, int]]:
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
    choices = policy(input_fidelities, fidelity_threshold, model)
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


class StrategyType(Enum):
    DIRECT=auto()
    OPT_SEARCH=auto()
    DAG=auto()

@dataclass
class Strategy:
    name: str
    type: StrategyType
    # exactly one of the two below is set, depending on `type`
    policy: PolicyFunction | None = None
    action_generator_factory: Callable[[list[tuple[str, float]], PurificationModel], ActionsGenerator] | None = None




THRESHOLD = 0.925
MIN_FIDELITY = 0.8
MAX_FIDELITY = 0.925
CONFIG_NAME = f"WERNER {MIN_FIDELITY} -> {THRESHOLD}"
MODEL = PurificationModel.WERNER
NUM_SAMPLES = 1000
MAX_PAIRS = 50
AVG_TIME_CUTOFF = 0.02

STRATEGIES: list[Strategy] = [
    Strategy("DAG all_possible_actions", StrategyType.DAG, action_generator_factory=lambda ignored1, ignored2: generate_all_possible_actions),
    Strategy("DAG all_single_pair", StrategyType.DAG, action_generator_factory=lambda ignored1, ignored2: generate_single_pair_actions),
    Strategy("DAG single pair inertia", StrategyType.DAG, action_generator_factory=lambda ignored1, ignored2: generate_single_pair_actions_inertia),
    Strategy("DAG highest fid single choice", StrategyType.DAG, action_generator_factory=get_highest_fid_single_choice_generator),
    Strategy("DAG lowest fid single choice", StrategyType.DAG, action_generator_factory=get_lowest_fid_single_choice_generator),
    Strategy("DAG sorted_fid_increment", StrategyType.DAG, action_generator_factory=get_sorted_fid_increment_generator),
    Strategy("DAG sorted_fid", StrategyType.DAG, action_generator_factory=get_sorted_fid_generator),
    Strategy("DAG sorted_increment", StrategyType.DAG, action_generator_factory=get_sorted_increment_generator),
    Strategy("DIRECT single pair highest fid", StrategyType.DIRECT, policy=single_pair_greedy_policy_highest),
    Strategy("DIRECT single pair lowest fid", StrategyType.DIRECT, policy=single_pair_greedy_policy_lowest),
    Strategy("DIRECT single pair highest deltaF", StrategyType.DIRECT, policy=bit_flip_highest_deltaF_single_choice_policy),
    Strategy("DIRECT all pairs opposite fid (middle)", StrategyType.DIRECT, policy=all_pairs_policy_opposite_middle_hole),
    Strategy("DIRECT all pairs opposite fid (head)", StrategyType.DIRECT, policy=all_pairs_policy_opposite_head_hole),
    Strategy("DIRECT all pairs opposite fid (tail)", StrategyType.DIRECT, policy=all_pairs_policy_opposite_tail_hole),
    Strategy("OPT_SEARCH highest fid", StrategyType.OPT_SEARCH, policy=get_optimistic_search_policy(choose_tree_highest_fid)),
    Strategy("OPT_SEARCH lowest fid", StrategyType.OPT_SEARCH, policy=get_optimistic_search_policy(choose_tree_lowest_fid)),
    Strategy("OPT_SEARCH most choices highest fid", StrategyType.OPT_SEARCH, policy=get_optimistic_search_policy(choose_tree_most_choices_highest_fid)),
    Strategy("OPT_SEARCH most choices lowest fid", StrategyType.OPT_SEARCH, policy=get_optimistic_search_policy(choose_tree_most_choices_lowest_fid)),
]



def stateless_sim(sample_i: int, seed: int, num_pairs: int, min_fid: float, max_fid: float, threshold: float, model: PurificationModel, strategy_index: int) -> tuple[int, int, float, float, float]: # (strat_i, sample_i, avg_usable, avg_steps, time)
    sim_start_time: float = time.time()
    strategy: Strategy =STRATEGIES[strategy_index]
    strategy_type: StrategyType = strategy.type
    strategy_policy: PolicyFunction | None = strategy.policy
    strategy_actions_generator_factory: Callable[[list[tuple[str, float]], PurificationModel], ActionsGenerator] | None = strategy.action_generator_factory

    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED is not set to 0"
    rng = np.random.default_rng(seed)
    def _input_generator() -> list[float]:
        to_return = sorted([rng.uniform(min_fid, max_fid) for _ in range(num_pairs)], reverse=True)
        return to_return
    input_fid_list: list[tuple[str, float]] = gen_initial_named_pairs(_input_generator)
    for f in cached_functions:
        f.cache_clear()
    if strategy_type == StrategyType.DAG:
        # here, how do I check that "policy_or_actions_generator_factory" is not a Policy?
        assert strategy_actions_generator_factory is not None
        a_g: ActionsGenerator = strategy_actions_generator_factory(input_fid_list, model)

        dag: PurificationDAG = PurificationDAG(input_fid_list, threshold, model, a_g)
        recursive_optimal_setup_main(dag)
        policy: PolicyFunction = PurificationDAGPolicy(dag)
        res = exact_recursive_simulation(policy, input_fid_list, threshold, model)
        assert np.allclose([average_usable_pairs_from_distribution(res), average_steps_from_distribution(res)], [dag.root.best_action_avg_usable,dag.root.best_action_avg_steps])
        usable: float = average_usable_pairs_from_distribution(res)
        steps: float = average_steps_from_distribution(res)
    elif strategy_type == StrategyType.DIRECT or strategy_type == StrategyType.OPT_SEARCH:
        assert strategy_policy is not None
        res = exact_recursive_simulation(strategy_policy, input_fid_list, threshold, model)
        usable: float = average_usable_pairs_from_distribution(res)
        steps: float = average_steps_from_distribution(res)
    else:
        exit(0)
    sim_stop_time: float = time.time()
    sim_time_s: float = sim_stop_time - sim_start_time
    return (strategy_index, sample_i, usable, steps, sim_time_s)

def progressive_increase_main() -> None:
    prog_start_time = time.time()


    num_pairs_range = list(range(2, MAX_PAIRS + 1))

    # https://claude.ai/share/e2d0a015-2561-4806-a8a2-da039242a93b
    # Flat/"tidy" list of rows: one row per (strategy, num_pairs, sample), each
    # row carrying the full context it was produced under. There's a lot of
    # repeated info across rows, but that keeps each row self-describing and
    # easy to filter/group later (e.g. with pandas) without needing the
    # strategies/config lists from elsewhere in the file.
    rows: list[dict[str, str | int | float]] = []

    active_strat_idxs: list[int] =list(range(len(STRATEGIES)))

    for _, num_pairs in enumerate(num_pairs_range):
        print(f"{num_pairs} PAIRS")

        params_iterable: list[tuple[int, int, int, float, float, float, PurificationModel, int]] = []

        for sample_i in range(NUM_SAMPLES):
            for strat_index in active_strat_idxs:
                strategy = STRATEGIES[strat_index]
                params_iterable.append((sample_i, num_pairs * NUM_SAMPLES + sample_i, num_pairs, MIN_FIDELITY, MAX_FIDELITY, THRESHOLD, MODEL, strat_index))
        # np.random.default_rng(0).shuffle(params_iterable)

        times_per_strat: dict[int, list[float]] = defaultdict(list)

        with ProcessPoolExecutor() as pool:
            results = pool.map(stateless_sim,
                                *zip(*params_iterable), # transposes single list of tuples into 8 lists, one per argument
                                chunksize=NUM_SAMPLES//50,
                            )
            for result in results:
                ret_strat_index, ret_sample_index, usable, steps, duration_s = result
                strategy = STRATEGIES[ret_strat_index]
                rows.append({
                    "config_name": CONFIG_NAME,
                    "model": MODEL.name,
                    "min_fidelity": MIN_FIDELITY,
                    "max_fidelity": MAX_FIDELITY,
                    "threshold": THRESHOLD,
                    "strategy_name": strategy.name,
                    "strategy_type": strategy.type.name,
                    "num_pairs": num_pairs,
                    "sample_i": ret_sample_index,
                    "usable": usable,
                    "steps": steps,
                    "duration_s": duration_s,
                })
                times_per_strat[ret_strat_index].append(duration_s)

        for k in times_per_strat.keys():
            assert len(times_per_strat[k]) == NUM_SAMPLES
            avg_time: float = sum(times_per_strat[k]) / len(times_per_strat[k])
            if avg_time >= AVG_TIME_CUTOFF:
                assert k in active_strat_idxs
                active_strat_idxs.remove(k)
                assert k not in active_strat_idxs
                print(f"{STRATEGIES[k].name} killed (avg {avg_time} s)")
        if len(active_strat_idxs) == 0:
            print(f"AVG_TIME_CUTOFF ({AVG_TIME_CUTOFF} s) reached for all strategies")
            break

    prog_end_time = time.time()
    print(f"Total execution time: {prog_end_time - prog_start_time} s")

    # --- Save results to disk for the plotting script ---
    out_path = "sim_results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    progressive_increase_main()