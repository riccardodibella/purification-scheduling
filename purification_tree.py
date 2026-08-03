

from itertools import combinations
from dataclasses import dataclass, field


def bit_flip_channel_purif_ok_prob(fid1: float, fid2: float) -> float:
    return fid1 * fid2 + (1 - fid1) * (1 - fid2)


def bit_flip_channel_purif_res_fidelity(fid1: float, fid2: float) -> float:
    return (
        fid1 * fid2
        /
        (fid1 * fid2 + (1 - fid1) * (1 - fid2))
    )

@dataclass
class TreeNode:
    state: list[tuple[str,float]]
    probability: float
    action: tuple[str,str] | None = None
    outcome: str | None = None
    usable_pairs: int = 0
    children: list["TreeNode"] = field(default_factory=list)

def build_purification_tree(
    state: list[tuple[str,float]],
    threshold: float = 0.95,
    probability: float = 1.0,
    usable_pairs: int = 0,
) -> TreeNode:

    node = TreeNode(
        state=state.copy(),
        probability=probability,
        usable_pairs=usable_pairs,
    )


    # terminal condition
    if len(state) < 2:
        return node


    # choose every possible pair
    for i,j in combinations(range(len(state)),2):

        name1, f1 = state[i]
        name2, f2 = state[j]


        new_name = f"({name1}+{name2})"


        p_ok = bit_flip_channel_purif_ok_prob(
            f1,f2
        )

        f_new = bit_flip_channel_purif_res_fidelity(
            f1,f2
        )


        remaining = [
            state[k]
            for k in range(len(state))
            if k not in (i,j)
        ]


        #################################
        # FAILURE
        #################################

        fail_child = build_purification_tree(
            remaining,
            threshold,
            probability*(1-p_ok),
            usable_pairs
        )

        fail_child.action = (name1,name2)
        fail_child.outcome = (
            f"FAIL removing {name1},{name2}"
        )

        node.children.append(fail_child)



        #################################
        # SUCCESS
        #################################

        if f_new >= threshold:

            # becomes usable, not returned
            success_state = remaining

            success_child = build_purification_tree(
                success_state,
                threshold,
                probability*p_ok,
                usable_pairs+1
            )

            success_child.outcome = (
                f"SUCCESS {new_name} "
                f"= {f_new:.4f} USABLE"
            )


        else:

            # put purified pair back
            success_state = remaining + [
                (new_name,f_new)
            ]

            success_child = build_purification_tree(
                success_state,
                threshold,
                probability*p_ok,
                usable_pairs
            )


            success_child.outcome = (
                f"SUCCESS {new_name} "
                f"= {f_new:.4f} returned"
            )


        success_child.action = (
            name1,
            name2
        )

        node.children.append(success_child)



    return node




def print_tree(node: TreeNode, depth: int = 0):

    indent = "    " * depth

    state_string = [
        f"{name}:{fid:.4f}"
        for name, fid in node.state
    ]

    print(
        indent +
        f"STATE: {state_string} "
        f"| P={node.probability:.6f} "
        f"| usable={node.usable_pairs}"
    )


    # group children by action
    grouped = {}

    for child in node.children:

        if child.action not in grouped:
            grouped[child.action] = []

        grouped[child.action].append(child)


    for action, children in grouped.items():

        name1, name2 = action

        print(
            indent +
            f"│"
        )

        print(
            indent +
            f"├─ SELECT pair: {name1} + {name2}"
        )


        for child in children:

            print(
                indent +
                f"│  ├─ {child.outcome}"
            )

            print_tree(
                child,
                depth + 2
            )


if __name__=="__main__":

    initial_state = [
        ("a",0.70),
        ("b",0.75),
        ("c",0.80),
        ("d",0.81),
    ]


    root = build_purification_tree(
        initial_state,
        threshold=0.99
    )


    print_tree(root)