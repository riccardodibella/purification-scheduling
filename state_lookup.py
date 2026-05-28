# pyright: strict
from typing import Callable
from math import log10, ceil
import numpy as np

rng = np.random.default_rng(0)


PolicyFunction = Callable[[list[tuple[str, float]], float], list[tuple[int, int]]]

StateDescription = str
ChoiceDescription = str
lookup_dict: dict[StateDescription, ChoiceDescription] = {}
lookup_dict["0,1,2,3"]="0:1"

def sort_fid_named_list(l: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(l, key=lambda x: x[1], reverse=True)

def sort_str_named_list(l: list[tuple[str, float]]) -> list[tuple[str, float]]:
    # Lexicographic ascending order
    return sorted(l, key=lambda x: x[0], reverse=False)

def encode_state_description(l: list[tuple[str, float]]) -> StateDescription:
    l = sort_str_named_list(l)
    return ','.join([t[0] for t in l])


def decode_choice_description(s: ChoiceDescription) -> list[tuple[str, str]]:
    arr: list[str] = s.split(",")
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
    if input_state not in lookup_dict:
        print("Piva piva l'olio d'oliva")
        print(input_state)
        assert False
    choice_str: ChoiceDescription = lookup_dict[input_state]
    to_return = decode_choice(l, choice_str)
    return to_return

def gen_initial_pairs() -> list[float]:
    return [0.85, 0.8, 0.7, 0.6]

def gen_initial_named_pairs() -> list[tuple[str, float]]:
    fids: list[float] = gen_initial_pairs()
    fids = sorted(fids, reverse=True)
    num_chars = ceil(log10(len(fids)))
    to_return = [(f"{i}".zfill(num_chars), fids[i]) for i in range(len(fids))]
    return to_return

initial = gen_initial_named_pairs()
print(initial)
print(encode_state_description(initial))

choice = lookup_policy(initial, 0.9)
print(choice)