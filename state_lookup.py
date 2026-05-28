# pyright: strict
from typing import Callable
from math import log10, ceil
import numpy as np

rng = np.random.default_rng(0)


PolicyFunction = Callable[[list[tuple[str, float]], float], list[tuple[int, int]]]

StateDescription = str
ChoiceDescription = str
lookup_dict: dict[StateDescription, ChoiceDescription] = {}

def sort_fid_named_list(l: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(l, key=lambda x: x[1], reverse=True)

def sort_str_named_list(l: list[tuple[str, float]]) -> list[tuple[str, float]]:
    # Lexicographic ascending order
    return sorted(l, key=lambda x: x[0], reverse=False)

def get_state_descr(l: list[tuple[str, float]]) -> StateDescription:
    l = sort_str_named_list(l)
    return f"[{','.join([t[0] for t in l])}]"


def gen_initial_pairs() -> list[float]:
    return [0.8, 0.7, 0.6]

def gen_initial_named_pairs() -> list[tuple[str, float]]:
    fids: list[float] = gen_initial_pairs()
    fids = sorted(fids, reverse=True)
    num_chars = ceil(log10(len(fids)))
    to_return = [(f"{i}".zfill(num_chars), fids[i]) for i in range(len(fids))]
    return to_return

initial = gen_initial_named_pairs()
print(initial)
print(get_state_descr(initial))