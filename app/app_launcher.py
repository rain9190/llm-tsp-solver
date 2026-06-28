"""
TSP Solver — Gradio interface for the fine-tuned Qwen2.5-1.5B model
(SFT + reward, round-1 merged), with best-of-8 decoding.

HOW TO RUN (in Google Colab, T4 GPU runtime):

CELL 1 : INSTALL THE FOLLOWING DEPENDENCIES
    !pip install -q -U "transformers>=4.46" accelerate bitsandbytes "protobuf>=5.26,<6" gradio
    !pip uninstall -q -y torchao
The model needs a GPU, so run this in Colab — not on a CPU-only laptop.

CELL 2 : MOUNT YOUR GOOGLE DRIVE WHERE THE FINE-TUNED ADAPTER IS DOWNLOADED
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    print(os.path.exists('/content/drive/MyDrive/llm-tsp/reward-adapter-1.5b-r1-merged'))
    print(os.listdir('/content/drive/MyDrive/llm-tsp'))
    
Proceed with the code below in CELL 3 if CELL 2 outputs "True"
"""
import torch, numpy as np, re, gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- model: the merged round-1 reward model (loads directly, no adapter) ----
MODEL_PATH = "/content/drive/MyDrive/llm-tsp/reward-adapter-1.5b-r1-merged"

SYSTEM_PROMPT = (
    "You are an expert solver for the Travelling Salesman Problem. "
    "Given city coordinates, output the shortest closed tour that visits every "
    "city exactly once. Reply with ONLY the visiting order as a space-separated "
    "list of city indices, starting from city 0, wrapped in <tour> and </tour> tags.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# ---- helpers ----
def make_prompt(coords):
    lines = [f"City {i}: ({x:.3f}, {y:.3f})" for i, (x, y) in enumerate(coords)]
    return (f"Solve this TSP instance with {len(coords)} cities.\n"
            + "\n".join(lines) + "\nReturn the tour.")

def parse_tour(text, n):
    m = re.search(r"<tour>(.*?)</tour>", text, re.DOTALL)
    nums = re.findall(r"\d+", m.group(1) if m else text)
    return [int(x) for x in nums] if nums else None

def is_feasible(tour, n):
    return tour is not None and sorted(tour) == list(range(n))

def tour_length(coords, tour):
    pts = coords[tour]; nxt = np.roll(pts, -1, axis=0)
    return float(np.sqrt(((pts - nxt) ** 2).sum(axis=1)).sum())

@torch.no_grad()
def _generate(coords):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(coords)}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=128, do_sample=True,
                         temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

def best_of_8(coords):
    n = len(coords); best, best_len = None, float("inf")
    for _ in range(8):
        tour = parse_tour(_generate(coords), n)
        if is_feasible(tour, n):
            L = tour_length(coords, tour)
            if L < best_len:
                best, best_len = tour, L
    return best, best_len

# ---- UI logic ----
def solve(n_cities, coord_text):
    n = int(n_cities)
    # parse "x, y" per line
    rows = []
    for line in coord_text.strip().splitlines():
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) >= 2:
            rows.append([float(nums[0]), float(nums[1])])
    if len(rows) != n:
        return (f"You said {n} cities but entered {len(rows)} coordinate pairs. "
                f"Please enter exactly {n} lines of 'x, y'."), None
    coords = np.array(rows)
    tour, L = best_of_8(coords)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c="#1f4e79", s=60, zorder=3)
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(6, 6))
    if tour is not None:
        loop = tour + [tour[0]]
        ax.plot(coords[loop, 0], coords[loop, 1], c="#2e7d32", lw=1.8, zorder=2)
        msg = f"Tour (visiting order): {tour}\nLength: {L:.4f}"
    else:
        msg = "No valid tour found — try again."
    ax.set_title("TSP tour"); ax.set_aspect("equal")
    return msg, fig

with gr.Blocks(title="LLM TSP Solver") as app:
    gr.Markdown("# LLM TSP Solver\n"
                "Fine-tuned **Qwen2.5-1.5B** (SFT + reward) solving the Travelling "
                "Salesman Problem end-to-end")
    n_in = gr.Number(label="How many cities?", value=8, precision=0)
    coord_in = gr.Textbox(
        label="Enter the coordinates — one 'x, y' per line (values 0–1)",
        placeholder="0.10, 0.40\n0.80, 0.20\n0.50, 0.90\n...", lines=8)
    btn = gr.Button("Solve", variant="primary")
    out_txt = gr.Textbox(label="Result")
    out_plot = gr.Plot(label="Tour")
    btn.click(solve, [n_in, coord_in], [out_txt, out_plot])

app.launch(share=True)