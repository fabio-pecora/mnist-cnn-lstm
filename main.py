# main.py
import argparse, os, random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from download_mnist import get_dataloaders
from cnn import SimpleCNN
from lstm import RowLSTM

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def acc(logits, y):
    return (logits.argmax(1) == y).sum().item(), y.size(0)

def train_one_epoch(model, loader, crit, opt, device):
    model.train(); tot_loss = 0; ok = 0; n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); out = model(x); loss = crit(out, y); loss.backward(); opt.step()
        tot_loss += loss.item() * x.size(0); c, t = acc(out, y); ok += c; n += t
    return tot_loss / n, ok / n

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval(); tot_loss = 0; ok = 0; n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x); loss = crit(out, y)
        tot_loss += loss.item() * x.size(0); c, t = acc(out, y); ok += c; n += t
    return tot_loss / n, ok / n

def build_model(args):
    if args.network == "cnn":
        return SimpleCNN(dropout_p=args.dropout)
    if args.network == "lstm":
        return RowLSTM(
            input_size=28,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout_p=args.dropout,
            bidirectional=args.bidirectional,
            num_classes=10
        )
    raise ValueError("Unknown network")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", choices=["cnn","lstm"], default="cnn")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["adam","sgd"], default="adam")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--hidden_size", type=int, default=128)   # for LSTM
    p.add_argument("--num_layers", type=int, default=1)      # for LSTM
    p.add_argument("--bidirectional", action="store_true")   # for LSTM
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Windows-safe DataLoader settings
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.bs, num_workers=0, pin_memory=False
    )

    model = build_model(args).to(device)
    print(model)

    crit = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_v = float("inf"); best = None; patience = 3; noimp = 0
    for ep in range(1, args.epochs + 1):
        trL, trA = train_one_epoch(model, train_loader, crit, opt, device)
        vaL, vaA = evaluate(model, val_loader, crit, device)
        print(f"Epoch {ep:02d}  train {trL:.4f}  {trA:.4f}  val {vaL:.4f}  {vaA:.4f}")
        if vaL < best_v - 1e-4:
            best_v = vaL; best = model.state_dict(); noimp = 0
        else:
            noimp += 1
            if noimp >= patience:
                print("Early stopping"); break

    if best is not None: model.load_state_dict(best)

    trL, trA = evaluate(model, train_loader, crit, device)
    vaL, vaA = evaluate(model, val_loader, crit, device)
    teL, teA = evaluate(model, test_loader, crit, device)
    print("Final results")
    print(f"Train  loss {trL:.4f}  acc {trA:.4f}  error {1 - trA:.4f}")
    print(f"Val    loss {vaL:.4f}  acc {vaA:.4f}  error {1 - vaA:.4f}")
    print(f"Test   loss {teL:.4f}  acc {teA:.4f}  error {1 - teA:.4f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), f"outputs/{args.network}_best.pt")

if __name__ == "__main__":
    main()
