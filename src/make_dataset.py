import json, numpy as np
from tsp_env import generate_instance, solve_with_ortools, tour_length
from serialize import make_prompt, make_answer, SYSTEM_PROMPT

def build_split(n_samples, n_min=10, n_max=20, seed0=0):
    rows = []
    for k in range(n_samples):
        n = int(np.random.default_rng(seed0 + k).integers(n_min, n_max + 1))
        coords = generate_instance(n, seed=seed0 + k)
        tour = solve_with_ortools(coords)
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_prompt(coords)},
                {"role": "assistant", "content": make_answer(tour)},
            ],
            "coords": coords.tolist(), 
            "opt_len": tour_length(coords, tour),
            # kept so eval can recompute lengths
        })
    return rows
    
def save_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    train = build_split(2000, seed0=0) # ~3-4 minutes on your Mac
    test = build_split(200, seed0=10_000) # disjoint seeds = unseen instances
    save_jsonl(train, "../data/train.jsonl")
    save_jsonl(test, "../data/test.jsonl")
    print(f"Saved {len(train)} train and {len(test)} test examples.")