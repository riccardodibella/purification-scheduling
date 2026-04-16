# pyright: basic


# In general, some other parameters may be useful: iteration_number: int, ready_pairs_count: int
# We ignore those for now, since we don't use them
def single_pair_greedy_policy(l: list[float]) -> list[tuple[int, int]]:
    if(len(l) < 2):
        return []
    working_l = zip(l, list(range(len(l))))
    working_l = sorted(working_l, key=lambda x: x[0], reverse=True) # descending order (higher fidelity first)
    return [(working_l[0][1],working_l[1][1])]

def purify_sample(fidelities: list[float], choices: list[tuple[int, int]]) -> list[float]:
    return []

if __name__ == "__main__":
    a_list = [0.67, 0.89, 0.56]
    choice = single_pair_greedy_policy(a_list)
    print(choice)
    new_list = purify_sample(a_list, choice)
    