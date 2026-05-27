# pyright: strict
from typing import Callable

import numpy as np

rng = np.random.default_rng(0)


PolicyFunction = Callable[[list[tuple[str, float]], float], list[tuple[int, int]]]

StateDescription = str
ChoiceDescription = str
lookup_dict: dict[StateDescription, ChoiceDescription] = {}

def get_state_descr(l: list[tuple[str, float]]) -> StateDescription:

    return ""


def gen_initial_pairs() -> list[float]:
    return [0.8, 0.7, 0.6]

def gen_initial_named_pairs() -> list[tuple[str, float]]:
    fids: list[float] = gen_initial_pairs()
    fids = sorted(fids, reverse=True)
    to_return = [(f"{i}", fids[i]) for i in range(len(fids))]
    return to_return

print(gen_initial_named_pairs())