"""Train and verify the Adam optimizer reproduction on UCI Optdigits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import nn

SEEDS = (42, 43, 44)
GRID = {"sgd": [0.001, 0.003, 0.01, 0.03, 0.1],
        "adam": [0.0001, 0.0003, 0.001, 0.003, 0.01]}
EPOCHS = 20
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"

def network():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

class Adam:
    """Algorithm 1 with an optional bias-correction ablation."""
    def __init__(self, parameters, lr, corrected=True):
        self.params = list(parameters)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.lr, self.corrected, self.t = lr, corrected, 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m.mul_(0.9).add_(g, alpha=0.1)
            v.mul_(0.999).addcmul_(g, g, value=0.001)
            mh = m / (1 - 0.9 ** self.t) if self.corrected else m
            vh = v / (1 - 0.999 ** self.t) if self.corrected else v
            p.addcdiv_(mh, vh.sqrt().add_(1e-8), value=-self.lr)

def load(directory, suffix):
    file = directory / f"optdigits.{suffix}"
    if not file.exists():
        urlretrieve(URL + file.name, file)
    data = np.loadtxt(file, delimiter=",", dtype=np.int64)
    return (torch.tensor(data[:, :64], dtype=torch.float32) / 16,
            torch.tensor(data[:, 64]), hashlib.sha256(file.read_bytes()).hexdigest())

def split(y):
    """Fixed class-stratified validation subset from the official training file."""
    rng = np.random.default_rng(2026)
    indices = [rng.permutation(np.flatnonzero(y.numpy() == digit)) for digit in range(10)]
    train = np.concatenate([i[round(len(i)*0.2):] for i in indices])
    val = np.concatenate([i[:round(len(i)*0.2)] for i in indices])
    return train, val

@torch.inference_mode()
def evaluate(model, x, y):
    logits = model(x)
    pred = logits.argmax(1).numpy()
    return float(nn.functional.cross_entropy(logits, y)), pred

def fit(x, y, mode, lr, seed):
    torch.manual_seed(seed)
    model = network()
    optimizer = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                 if mode == "sgd" else Adam(model.parameters(), lr, corrected=mode == "adam"))
    generator = torch.Generator().manual_seed(seed + 1000)
    history = []
    for epoch in range(1, EPOCHS + 1):
        for batch in torch.randperm(len(y), generator=generator).split(128):
            model.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x[batch]), y[batch]).backward()
            optimizer.step()
        loss, pred = evaluate(model, x, y)
        history.append({"epoch": epoch, "loss": loss, "accuracy": float(np.mean(pred == y.numpy()))})
    return model, history

def check_adam():
    """Compare float64 updates with PyTorch Adam on fixed gradients."""
    p = nn.Parameter(torch.tensor([0.2, -0.6, 1.0], dtype=torch.float64))
    q = nn.Parameter(p.detach().clone())
    actual = Adam([p], 0.003)
    reference = torch.optim.Adam([q], lr=0.003, foreach=False)
    for grad in ([0.1,-0.3,0.7], [-0.2,0.5,0.0], [0.5,0.2,-0.4], [0.0,0.2,0.9]):
        p.grad = torch.tensor(grad, dtype=torch.float64)
        q.grad = p.grad.clone()
        actual.step()
        reference.step()
        torch.testing.assert_close(p, q, rtol=0, atol=1e-12)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.data.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    check_adam()
    xt, yt, test_hash = load(args.data, "tes")
    preds = {"label": yt.numpy()}
    if not args.verify:
        x, y, train_hash = load(args.data, "tra")
        tr, va = split(y)
        report = {"n_train": len(y), "n_test": len(yt), "tune_train": len(tr), "tune_val": len(va),
                  "train_sha256": train_hash, "test_sha256": test_hash, "torch": torch.__version__,
                  "tuning": {}, "runs": []}
        rates = {}
        for mode, candidates in GRID.items():
            scores = []
            for lr in candidates:
                acc = []
                for seed in SEEDS:
                    model, _ = fit(x[tr], y[tr], mode, lr, seed)
                    _, pred = evaluate(model, x[va], y[va])
                    acc.append(float(np.mean(pred == y[va].numpy())))
                scores.append({"lr": lr, "accuracy": acc})
            rates[mode] = max(scores, key=lambda row: np.mean(row["accuracy"]))["lr"]
            report["tuning"][mode] = scores
        rates["uncorrected"] = rates["adam"]
        report["rates"] = rates
        print("Selected rates:", rates, flush=True)
    for seed in SEEDS:
        for mode in ("sgd", "adam", "uncorrected"):
            name = f"{mode}_{seed}"
            weights = args.output / f"{name}.pt"
            if not args.verify:
                model, history = fit(x, y, mode, rates[mode], seed)
                torch.save(model.state_dict(), weights)
            model = network()
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            loss, pred = evaluate(model, xt, yt)
            preds[name] = pred
            if not args.verify:
                cm = np.zeros((10, 10), dtype=int)
                np.add.at(cm, (yt.numpy(), pred), 1)
                report["runs"].append({"seed": seed, "model": mode, "history": history,
                    "correct": int(cm.trace()), "accuracy": float(cm.trace()/len(yt)), "test_loss": loss,
                    "macro_f1": float(np.mean(2*cm.diagonal()/(cm.sum(0)+cm.sum(1)))),
                    "confusion": cm.tolist()})
    path = args.output / "predictions.csv"
    if args.verify:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            assert next(reader) == list(preds)
            np.testing.assert_array_equal(np.array(list(reader), dtype=int), np.column_stack(list(preds.values())))
        print(f"Verified {len(yt)} predictions for each of {len(preds)-1} checkpoints.")
    else:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(preds)
            writer.writerows(zip(*preds.values()))
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
