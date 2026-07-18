#!/usr/bin/env python
"""Fractus-1B cloud training script — GPU optimized, auto-push to HF.

Designed for RunPod/QuickPod RTX 4090 or A100.
Automatically uploads every checkpoint to HuggingFace.

Usage on cloud GPU:
    pip install torch datasets huggingface_hub
    python train_1b_cloud.py --epochs 20
    
Environment variables needed:
    HF_TOKEN=your_token_here
"""
import argparse, gc, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from fractus.model_1b import Fractus1B
from fractus.tokenizer import FractusTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/Fractus"


def upload_hf(path, repo_path):
    """Upload to HuggingFace. Never fails the training."""
    if not HF_TOKEN:
        print(f"  [HF] No token, skipping.", flush=True)
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=path, path_in_repo=repo_path,
                       repo_id=HF_REPO, repo_type="model")
        print(f"  [HF] Uploaded {repo_path}", flush=True)
    except Exception as e:
        print(f"  [HF] Failed: {type(e).__name__} — training continues.", flush=True)


def save_and_upload(model, optimizer, epoch, loss, acc, config, ckpt_dir):
    """Save checkpoint locally + upload to HF."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"fractus_1b_epoch{epoch}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "loss": loss,
        "accuracy": acc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [ckpt] {path} ({size_mb:.0f}MB)", flush=True)
    upload_hf(path, f"checkpoints/fractus_1b_epoch{epoch}.pt")
    # Also upload as "latest" so it's easy to find.
    upload_hf(path, "checkpoints/fractus_1b_latest.pt")
    # Delete old checkpoint to save disk on cloud.
    if epoch > 1:
        old = os.path.join(ckpt_dir, f"fractus_1b_epoch{epoch-1}.pt")
        if os.path.exists(old):
            os.remove(old)
            print(f"  [disk] Removed old checkpoint {old}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fractus-1B Cloud Training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64,
                       help="Longer seq = better context (GPU can handle it)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (GPU parallelism)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--corpus", type=str, default=None,
                       help="Path to corpus. If not found, builds it.")
    parser.add_argument("--resume", type=str, default=None,
                       help="Checkpoint to resume from")
    parser.add_argument("--upload-every", type=int, default=1,
                       help="Upload checkpoint every N epochs")
    args = parser.parse_args()

    # Detect device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)", flush=True)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected. Running on CPU.", flush=True)

    torch.manual_seed(42)
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    print(f"Threads: {num_threads}", flush=True)

    # Load or build corpus.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if args.corpus:
        corpus_path = args.corpus
    else:
        # Try communication corpus first, then ultimate, then mega.
        for name in ["communication_corpus.pt", "ultimate_corpus.pt", "mega_corpus.pt"]:
            p = os.path.join(project_dir, "data", name)
            if os.path.exists(p):
                corpus_path = p
                break
        else:
            # Build it.
            print("No corpus found. Building communication corpus...", flush=True)
            import subprocess
            subprocess.run([sys.executable, os.path.join(script_dir, "build_communication_corpus.py")], check=True)
            corpus_path = os.path.join(project_dir, "data", "communication_corpus.pt")

    print(f"Loading corpus: {corpus_path}", flush=True)
    tokens = torch.load(corpus_path, weights_only=False).long()
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Build model.
    print("Building Fractus-1B...", flush=True)
    model = Fractus1B(
        vocab_size=50257, d_model=768, n_layers=8, n_heads=12, d_head=64,
        n_levels=2, n_experts=64, top_k=2, expert_d_ff=1024, siren_rank=16,
        max_seq_len=args.seq_len,
    ).to(device)
    n = model.n_params()
    cap = model.n_effective_capacity()
    print(f"  Params: {n:,} ({n/1e6:.0f}M)", flush=True)
    print(f"  Capacity: {cap:,} ({cap/1e9:.2f}B)", flush=True)
    print(f"  RAM: {n*4/1e9:.1f}GB", flush=True)

    # Resume if specified.
    start_epoch = 0
    if args.resume:
        print(f"Resuming from: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch}, loss={ckpt.get('loss','?')}", flush=True)

    # Optimizer + scheduler.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    if start_epoch > 0:
        for _ in range(start_epoch):
            sched.step()

    # AMP for GPU.
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    tok = FractusTokenizer.gpt2_compatible()
    seq = args.seq_len
    batch_size = args.batch_size
    ckpt_dir = os.path.join(project_dir, "checkpoints")
    n_steps = len(tokens) // seq // batch_size

    print(f"\nTraining {args.epochs} epochs", flush=True)
    print(f"  seq_len={seq}, batch_size={batch_size}, lr={args.lr}", flush=True)
    print(f"  {n_steps:,} steps/epoch", flush=True)
    print(f"  AMP: {'ON' if use_amp else 'OFF'}", flush=True)
    print(f"  HF upload: every {args.upload_every} epochs → {HF_REPO}", flush=True)
    print("=" * 70, flush=True)

    initial_loss = None
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.perf_counter()
        ep_loss = 0.0
        ep_n = 0

        # Create batches.
        for batch_start in range(0, len(tokens) - seq * batch_size - 1, seq * batch_size):
            # Build batch.
            inp_list = []
            tgt_list = []
            for b in range(batch_size):
                offset = batch_start + b * seq
                inp_list.append(tokens[offset:offset + seq])
                tgt_list.append(tokens[offset + 1:offset + seq + 1])
            inp = torch.stack(inp_list).to(device)
            tgt = torch.stack(tgt_list).to(device)

            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits, aux = model(inp)
                    ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                    loss = ce + 0.001 * aux
                loss.backward()
            else:
                logits, aux = model(inp)
                ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                loss = ce + 0.001 * aux
  
