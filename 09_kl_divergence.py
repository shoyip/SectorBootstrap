import argparse
import csv
from pathlib import Path

import numpy as np


def _load_alignment(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D alignment in {path}, got shape {arr.shape}.")
        return arr.astype(np.int64, copy=False)

    raise ValueError(f"Unsupported alignment format in {path}.")


def _find_model_file(models_dir: Path, subaln_idx: int, step: int) -> Path:
    # Support both Step5 and Step05 naming conventions.
    patterns = [
        f"FullShuffling_SubAln{subaln_idx}_Step{step}*.npy",
        f"FullShuffling_SubAln{subaln_idx}_Step{step:02d}*.npy",
        f"FullShuffling_SubAln{subaln_idx}_Step{step:03d}*.npy",
        f"FullShuffling_SubAln{subaln_idx}_Step{step}/*.npy",
        f"FullShuffling_SubAln{subaln_idx}_Step{step:02d}/*.npy",
        f"FullShuffling_SubAln{subaln_idx}_Step{step:03d}/*.npy",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(sorted(models_dir.glob(pattern)))

    if not candidates:
        raise FileNotFoundError(
            f"No model file found for subaln={subaln_idx}, step={step} in {models_dir}."
        )

    return candidates[0]


def _resolve_train_file(train_dir: Path, subaln_idx: int) -> Path:
    candidates = [
        train_dir / f"subaln{subaln_idx}_seq.npy",
        train_dir / f"subaln_seq_{subaln_idx}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find training alignment for subalignment "
        f"{subaln_idx} in {train_dir}. Tried: "
        + ", ".join(str(p.name) for p in candidates)
    )


def _site_frequencies(aln: np.ndarray, q: int, pseudocount: float) -> np.ndarray:
    n_seq, length = aln.shape
    freqs = np.zeros((length, q), dtype=np.float64)

    for i in range(length):
        counts = np.bincount(aln[:, i], minlength=q).astype(np.float64)
        counts += pseudocount
        counts /= counts.sum()
        freqs[i] = counts

    if n_seq == 0:
        raise ValueError("Alignment has zero sequences.")

    return freqs


def _kl_per_site(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    # p and q are already smoothed, so no log(0) issues.
    return np.sum(p * np.log(p / q), axis=1)


def _append_csv_row(csv_file: Path, row: dict) -> None:
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_file.exists()
    fieldnames = [
        "subaln_idx",
        "step",
        "model_file",
        "n_train",
        "n_generated",
        "length",
        "q",
        "kl_generated_to_train_mean",
        "kl_train_to_generated_mean",
        "js_mean",
    ]

    with csv_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute KL divergence between generated and training alignments."
    )
    parser.add_argument("subaln_idx", type=int, help="Subalignment index (0-9).")
    parser.add_argument("step", type=int, help="Shuffling step.")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory containing FullShuffling model folders.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="./data/FullSubAln",
        help="Directory containing subalignment sequence .npy files.",
    )
    parser.add_argument(
        "--generated-file",
        type=str,
        default=None,
        help="Optional explicit .npy alignment for generated sequences.",
    )
    parser.add_argument(
        "--generated-key",
        type=str,
        default="Test",
        help="Key in model .npy dictionary containing generated alignment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/kl_divergence",
        help="Where per-job KL outputs are written.",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1e-9,
        help="Pseudocount for frequency smoothing.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = _resolve_train_file(train_dir, args.subaln_idx)
    train_aln = _load_alignment(train_file)

    model_file = _find_model_file(models_dir, args.subaln_idx, args.step)
    model_data = np.load(model_file, allow_pickle=True).item()

    if args.generated_file is not None:
        generated_aln = _load_alignment(Path(args.generated_file))
        generated_source = str(Path(args.generated_file))
    else:
        generated_aln = model_data.get(args.generated_key)
        generated_source = f"{model_file}:{args.generated_key}"
        if generated_aln is None:
            available = ", ".join(sorted(model_data.keys()))
            raise ValueError(
                f"Key '{args.generated_key}' not found (or is None) in model output. "
                f"Available keys: {available}. "
                "Pass --generated-file to provide an explicit generated alignment."
            )
        generated_aln = np.asarray(generated_aln)
        if generated_aln.ndim != 2:
            raise ValueError(
                f"Generated alignment at key '{args.generated_key}' is not 2D: "
                f"shape={generated_aln.shape}"
            )

    if train_aln.shape[1] != generated_aln.shape[1]:
        raise ValueError(
            "Training and generated alignments have different lengths: "
            f"{train_aln.shape[1]} vs {generated_aln.shape[1]}."
        )

    q = int(max(train_aln.max(), generated_aln.max()) + 1)
    p_gen = _site_frequencies(generated_aln, q=q, pseudocount=args.pseudocount)
    p_train = _site_frequencies(train_aln, q=q, pseudocount=args.pseudocount)

    kl_gen_to_train = _kl_per_site(p_gen, p_train)
    kl_train_to_gen = _kl_per_site(p_train, p_gen)
    m = 0.5 * (p_gen + p_train)
    js_per_site = 0.5 * (_kl_per_site(p_gen, m) + _kl_per_site(p_train, m))

    metrics = {
        "subaln_idx": args.subaln_idx,
        "step": args.step,
        "model_file": str(model_file),
        "generated_source": generated_source,
        "n_train": int(train_aln.shape[0]),
        "n_generated": int(generated_aln.shape[0]),
        "length": int(train_aln.shape[1]),
        "q": q,
        "kl_generated_to_train_mean": float(np.mean(kl_gen_to_train)),
        "kl_train_to_generated_mean": float(np.mean(kl_train_to_gen)),
        "js_mean": float(np.mean(js_per_site)),
    }

    per_job_file = output_dir / f"kl_subaln{args.subaln_idx}_step{args.step}.npz"
    np.savez(
        per_job_file,
        kl_generated_to_train_per_site=kl_gen_to_train,
        kl_train_to_generated_per_site=kl_train_to_gen,
        js_per_site=js_per_site,
        metrics=metrics,
    )

    summary_csv = output_dir / "kl_summary.csv"
    _append_csv_row(summary_csv, {k: metrics[k] for k in [
        "subaln_idx",
        "step",
        "model_file",
        "n_train",
        "n_generated",
        "length",
        "q",
        "kl_generated_to_train_mean",
        "kl_train_to_generated_mean",
        "js_mean",
    ]})

    print(f"Model file: {model_file}")
    print(f"Generated source: {generated_source}")
    print(
        "KL(gen||train) mean = "
        f"{metrics['kl_generated_to_train_mean']:.6e}"
    )
    print(
        "KL(train||gen) mean = "
        f"{metrics['kl_train_to_generated_mean']:.6e}"
    )
    print(f"JS mean = {metrics['js_mean']:.6e}")
    print(f"Saved per-site metrics to {per_job_file}")
    print(f"Updated summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
