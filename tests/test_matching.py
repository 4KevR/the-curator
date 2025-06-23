from src.backend.modules.helpers.matching import match_by_equals, match_by_key, match_by_tolerance

res = match_by_key([1, 3, 5], ["5", "7", "1", "9"], equals=(lambda x, y: str(x) == y), right_key=lambda x: int(x))
assert res == ([(1, "1"), (5, "5")], [3], ["7", "9"])


res = match_by_equals([1, 3, 5], ["5", "7", "1", "9"], lambda l, r: l == int(r))  # noqa E741
assert res == ([(1, "1"), (5, "5")], [3], ["7", "9"])

res = match_by_tolerance(
    left=["Banana", "Apple", "Orange"],
    right=["Banan", "Bananas", "Banana", "Apfel"],
    tolerance_function=lambda l, r: l[0:2] == r[0:2],  # noqa E741
)
assert res == ([(["Banana"], ["Banan", "Bananas", "Banana"]), (["Apple"], ["Apfel"])], ["Orange"], [])

res = match_by_tolerance(
    left=["abc", "bcd", "cde", "ec"],
    right=["B", "C", "F"],
    tolerance_function=lambda l, r: len(set(l) & set(r.lower())) != 0,  # noqa E741
)
assert res == ([(["abc", "bcd", "cde", "ec"], ["B", "C"])], [], ["F"])
