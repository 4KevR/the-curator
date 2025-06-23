from typing import Any, Callable, TypeVar

LEFT = TypeVar("LEFT")
RIGHT = TypeVar("RIGHT")


# O(max(n, m))
# if multiple items in left/right have the same key, they are compared using the given equality function.
def match_by_key(
    left: list[LEFT],
    right: list[RIGHT],
    equals: Callable[[LEFT, RIGHT], bool],
    left_key: Callable[[LEFT], Any] = lambda x: x,
    right_key: Callable[[RIGHT], Any] = lambda x: x,
) -> tuple[list[tuple[LEFT, RIGHT]], list[LEFT], list[RIGHT]]:
    """
    Matches elements in left and right by key.
    If multiple items in left/right have the same key, they are compared using the given equality function.
    If all keys are unique, the function runs in O(n + m) time, else it can degenerate to match_by_equals with
    time complexity O(n * m) (if all keys are equal).

    Example:
       match_by_key([1, 3, 5], ["5", "7", "1", "9"], equals=(lambda x, y: str(x) == y), right_key=lambda x: int(x))
         returns
       ([(1, '1'), (5, '5')], [3], ['7', '9'])
    """
    left_by_key = dict()
    for l_val in left:
        l_key = left_key(l_val)
        if l_key not in left_by_key:
            left_by_key[l_key] = [l_val]
        else:
            left_by_key[l_key].append(l_val)

    right_by_key = dict()
    for r_val in right:
        r_key = right_key(r_val)
        if r_key not in right_by_key:
            right_by_key[r_key] = [r_val]
        else:
            right_by_key[r_key].append(r_val)

    all_keys = left_by_key.keys() | right_by_key.keys()

    match, only_left, only_right = [], [], []
    for key in all_keys:
        left_candidates = left_by_key.get(key, [])
        right_candidates = right_by_key.get(key, [])
        (tmp_match, tmp_only_left, tmp_only_right) = match_by_equals(left_candidates, right_candidates, equals)
        match.extend(tmp_match)
        only_left.extend(tmp_only_left)
        only_right.extend(tmp_only_right)

    return match, only_left, only_right


# O( n * m )
def match_by_equals(
    left: list[LEFT], right: list[RIGHT], equals: Callable[[LEFT, RIGHT], bool], allow_multiple_matches: bool = True
) -> tuple[list[tuple[LEFT, RIGHT]], list[LEFT], list[RIGHT]]:
    """
    Matches elements in left and right by equality.
    Takes O(n * m) time.
    If allow_multiple_matches is False, and a left/right element has multiple matches in the other collection, a
    ValueError is thrown.

    Example:
       match_by_equals([1, 3, 5], ["5", "7", "1", "9"], lambda l, r: l == int(r))
         returns
       ([(1, '1'), (5, '5')], [3], ['7', '9'])

    """
    matches = []
    left_matched = len(left) * [False]
    right_matched = len(right) * [False]

    for l_idx, l in enumerate(left):
        for r_idx, r in enumerate(right):
            if equals(l, r):
                if left_matched[l_idx]:
                    if allow_multiple_matches:
                        continue
                    raise ValueError(f"Left element #{l_idx}: {l} has multiple matches.")
                if right_matched[r_idx]:
                    if allow_multiple_matches:
                        continue
                    raise ValueError(f"Right element #{r_idx}: {r} has multiple matches.")
                left_matched[l_idx] = True
                right_matched[r_idx] = True
                matches.append((l, r))
    only_left = [l for l_idx, l in enumerate(left) if not left_matched[l_idx]]
    only_right = [r for r_idx, r in enumerate(right) if not right_matched[r_idx]]

    return matches, only_left, only_right


def match_by_tolerance(left: list[LEFT], right: list[RIGHT], tolerance_function: Callable[[LEFT, RIGHT], bool]):
    """
    Matches left and right entries by tolerance relation (equality relation that is not necessarily transitive).

    Example:
        left = ["Banana", "Apple", "Orange"]
        right = ["Banan", "Bananas", "Banana", "Apfel"]
        tolerance_function = lambda l, r: l[0:2] == r[0:2]

        returns
        ```
        [(["Banana"], ["Banan", "Bananas", "Banana"]), (["Apple"], ["Apfel"])], ["Orange"], []
        ```

    """

    # get all right ids that fit a left id and get all left ids that fit a given right id
    left_to_right = {l_key: [] for l_key, l in enumerate(left)}
    right_to_left = {r_key: [] for r_key, r in enumerate(right)}

    for l_key, l in enumerate(left):
        for r_key, r in enumerate(right):
            if tolerance_function(l, r):
                left_to_right[l_key].append(r_key)
                right_to_left[r_key].append(l_key)

    # match
    matches: list[tuple[list[LEFT], list[RIGHT]]] = []
    remaining_l_key = set(range(len(left)))
    remaining_r_key = set(range(len(right)))

    while len(remaining_l_key) > 0:
        l_keys_in_match = set()
        r_keys_in_match = set()

        l_key_queue = {next(iter(remaining_l_key))}
        r_key_queue = set()

        while len(l_key_queue) > 0 or len(r_key_queue) > 0:
            if len(l_key_queue) > 0:
                l_key = l_key_queue.pop()
                l_keys_in_match.add(l_key)
                remaining_l_key.remove(l_key)

                for r_key in left_to_right[l_key]:
                    if r_key in remaining_r_key:
                        r_key_queue.add(r_key)

            if len(r_key_queue) > 0:
                r_key = r_key_queue.pop()
                r_keys_in_match.add(r_key)
                remaining_r_key.remove(r_key)

                for l_key in right_to_left[r_key]:
                    if l_key in remaining_l_key:
                        l_key_queue.add(l_key)

        matches.append(([left[it] for it in sorted(l_keys_in_match)], [right[it] for it in sorted(r_keys_in_match)]))

    # no l_keys remaining. Remaining r_keys have no match
    for r_key in remaining_r_key:
        matches.append(([], [right[r_key]]))

    # find and separate empty matches
    true_matches: list[tuple[list[LEFT], list[RIGHT]]] = []
    only_left: list[LEFT] = []
    only_right: list[RIGHT] = []

    for left, right in matches:
        if len(left) == 0:
            assert len(right) == 1
            only_right.append(right[0])
        elif len(right) == 0:
            assert len(left) == 1
            only_left.append(left[0])
        else:
            true_matches.append((left, right))

    return true_matches, only_left, only_right
