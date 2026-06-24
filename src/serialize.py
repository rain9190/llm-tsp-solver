import re

SYSTEM_PROMPT = (
    "You are an expert solver for the Travelling Salesman Problem. "
    "Given city coordinates, output the shortest closed tour that visits every "
    "city exactly once. Reply with ONLY the visiting order as a space-separated "
    "list of city indices, starting from city 0, wrapped in <tour> and </tour> tags."
)

def make_prompt(coords):
    lines = [f"City {i}: ({x:.3f}, {y:.3f})" for i, (x, y) in enumerate(coords)]
    return (f"Solve this TSP instance with {len(coords)} cities.\n" + "\n".join(lines) + "\nReturn the tour.")

def make_answer(tour):
    return "<tour>" + " ".join(str(c) for c in tour) + "</tour>"

def parse_tour(text, n_cities):
    """Pull a tour out of model output. Returns list[int] or None."""
    m = re.search(r"<tour>(.*?)</tour>", text, re.DOTALL)
    nums = re.findall(r"\d+", m.group(1) if m else text)
    return [int(x) for x in nums] if nums else None

def is_feasible(tour, n_cities):
    """A tour is feasible iff it is a permutation of 0..n_cities-1."""
    return tour is not None and sorted(tour) == list(range(n_cities))