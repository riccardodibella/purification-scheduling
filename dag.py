# pyright: strict
from __future__ import annotations
from enum import Enum, auto
from itertools import chain, combinations, product
from typing import Callable
import numpy as np
import heapq
import math
import time
from functools import lru_cache # pyright: ignore[reportUnusedImport]

"""
# pyright: basic
from line_profiler import profile # PYTHONHASHSEED=0 kernprof -l -v dag.py
"""

import os
import sys

sys.set_int_max_str_digits(1_000_000)

if os.environ.get("PYTHONHASHSEED") != "0":
    print("Restarting and setting hash seed")
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

rng = np.random.default_rng(0)

PolicyFunction = Callable[[list[tuple[str, float]], float], list[tuple[int, int]]]

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

def bitstrings(n: int) -> list[list[bool]]:
    """
    Returns a list of all possible bitstrings ( = lists of bools ) of length n
    """
    return [list(bits) for bits in product([False, True], repeat=n)]

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


def gen_initial_pairs() -> list[float]:
    return [0.85, 0.8, 0.72, 0.7, 0.6]

def gen_initial_named_pairs() -> list[tuple[str, float]]:
    fids: list[float] = gen_initial_pairs()
    fids = sorted(fids, reverse=True)
    num_chars = math.ceil(math.log10(len(fids)))
    to_return = [(f"{i}".zfill(num_chars), fids[i]) for i in range(len(fids))]
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




class ActionItem:
    state_string: StateDescription
    choice: ChoiceDescription

    # The first element is the bitstring (list of bools) associated with the outcome for that children
    # The second element is the probability of having this outcome
    # The third element is the number of usable pairs generated in that transition
    # The fourth element is the child node
    resulting_children: list[ tuple[ list[bool], float, int , DAGNode ] ]
    def __init__(self, state_string: StateDescription, choice: ChoiceDescription, resulting_children: list[ tuple[ list[bool], float, int , DAGNode ] ]) -> None:
        self.state_string = state_string
        self.choice = choice
        self.resulting_children = resulting_children

class DAGNode:
    # Topological info
    parents: set[DAGNode]
    state_string: StateDescription # str
    actions: list[ActionItem]
    actions_generated: bool # used as a safety check to ensure that we visit each node only once when we build the DAG structure

    # Search info
    best_action_chosen: bool
    chosen_action_index: int
    best_action_avg_usable: float
    best_action_avg_steps: float

    def __init__(self, state_string: StateDescription) -> None:
        self.parents = set() # filled by the main DAG object
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

        self.root = self.add_node(node_state_string=self.entry_point_string, parent=None) # bootstrap the construction process

        if actions_generator is not None:
            self.construct_DAG(actions_generator)

    def add_node(self, node_state_string: StateDescription, parent: StateDescription | DAGNode | None) -> DAGNode:
        # This function is a no-op if we are adding a node with the same parent (possibly None) multiple times

        if node_state_string not in self.nodes_dict:
            node = DAGNode(node_state_string)
            self.nodes_dict[node_state_string] = node
        else:
            node = self.nodes_dict[node_state_string]
        assert node is not None
        if parent is not None:
            if isinstance(parent, StateDescription):
                assert parent in self.nodes_dict
                parent = self.nodes_dict[parent]
            assert isinstance(parent, DAGNode)
            node.parents.add(parent)
        return node

    def construct_DAG(self, actions_generator: ActionsGenerator) -> None:
        assert self.root is not None
        assert self.root.actions_generated is False

        hq: list[tuple[int, str, str | None]] = [] # heap with the nodes to be evaluated
        def _priority(s: StateDescription) -> int:
            return -1 * len(s.split(',')) # sort based on how many different inputs are there in the state
        heapq.heappush(hq, (_priority(self.entry_point_string), self.entry_point_string, None))
        while len(hq) != 0:
            _, current_state_string, parent_state_string = heapq.heappop(hq)
            parent_node: DAGNode | None = self.nodes_dict[parent_state_string] if parent_state_string is not None else None
            current_node: DAGNode = self.add_node(node_state_string=current_state_string, parent=parent_node)

            # We must do this check after doing add_node in order to always register the new parent of this node,
            # regardless of whether we already visited it and generated its actions or not
            if current_node.actions_generated:
                continue # We have already visited this node and generated its actions: nothing else to do, go to the next node in hq
            
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
                        input_fids: tuple[float, ...] = tuple([get_key_fidelity_recursive(k, self.initial_pairs, self.model) for k in input_keys])
                        assert len(input_fids) == 2
                        success_probability: float = purif_ok_prob(self.model, input_fids[0], input_fids[1])
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
                    new_node: DAGNode = self.add_node(node_state_string=outcome_result_str, parent=current_node)
                    resulting_children.append((bstring, outcome_probability, generated_usable_pairs, new_node))
                    # We add the new node to hq regardless of wheter its actions are already generated or not, because we still need to add the current node to its parents
                    heapq.heappush(hq, (_priority(new_node.state_string), new_node.state_string, current_node.state_string))
                if action_string == "":
                    assert len(resulting_children) == 0
                ai = ActionItem(current_state_string, action_string, resulting_children)
                current_node.add_action(ai)
                
            current_node.actions_generated = True
        print("construct_DAG finished")

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
        if (avg_usable > best_avg_usable) or ((avg_usable == best_avg_usable) and (avg_steps < best_avg_steps)):
            best_avg_usable = avg_usable
            best_avg_steps = avg_steps
            best_action_index = action_index
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
        assert node.actions[action_index].state_string == node.state_string
        choice_str: ChoiceDescription = node.actions[action_index].choice
        to_return = decode_choice(l, choice_str)
        return to_return
    















































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
    
def filter_usable_pairs(pairs: list[tuple[str, float]], threshold: float) -> tuple[int, list[tuple[str, float]]]:
    remaining_pairs = [p for p in pairs if p[1] < threshold]
    usable_counter = len(pairs) - len(remaining_pairs)
    return usable_counter, remaining_pairs

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
    dag: PurificationDAG = PurificationDAG(input_fid_list, threshold, model, generate_possible_actions)
    recursive_optimal_setup_main(dag)
    dag_policy = PurificationDAGPolicy(dag)
    for policy in [dag_policy]:
            end_distribution = exact_recursive_simulation(policy, input_fid_list, threshold, model)
            print(f"{policy.__name__}: {average_usable_pairs_from_distribution(end_distribution)} ({average_steps_from_distribution(end_distribution)} steps)")
    prog_end_time = time.time()
    print(f"Total execution time: {prog_end_time - prog_start_time} s")