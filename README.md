# AdaDCL

Code for **AdaDCL: Adaptive Noise Enhanced Diffusion Contrastive Learning for Recommendation**, Neurocomputing (2026).

[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231226020424)

## Installation

Use Python 3.11 and install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

The paper uses PyTorch 2.2.2 with CUDA 12.1. Install a PyTorch build compatible with your GPU when using a different machine.

## Data

Data and pretrained embeddings are supplied separately. Prepare the following structure for `office`, `ml-1m`, or `yelp`:

```text
data/
  office/
    train.txt
    test.txt
    uEmbeds_pt.npy
    iEmbeds_pt.npy
    item_social_emb.npy
    item_knowledge_emb.npy
```

Each interaction file contains two whitespace-separated integer columns: `user_id item_id`. IDs start at zero and must match the embedding row indices. User/item embeddings have shape `[n_users, 64]` and `[n_items, 64]`; both auxiliary item arrays have shape `[n_items, 64]`.

The two auxiliary arrays and normalized graph cache can be generated automatically if absent. Auxiliary generation can be expensive for large datasets. Rebuild caches when changing the split or pretrained embeddings. The current training code does not read `valid.txt`.

## Run

Run from this directory. CUDA is used when available; `--cuda 0` selects the first GPU. Logs and curves go to `log/`, and checkpoints go to `result/`.

```bash
# Amazon-Office
python -X utf8 main.py --data office --lr 0.005 --batch_size 2048 --timesteps 100 --beta 0.001

# MovieLens-1M
python -X utf8 main.py --data ml-1m --lr 0.005 --batch_size 8192 --timesteps 200 --bpr_alpha 0.9 --recon_weight 0.1 --beta 0.0001

# Yelp
python -X utf8 main.py --data yelp --lr 0.01 --batch_size 8192 --timesteps 200 --bpr_alpha 0.9 --recon_weight 0.1 --beta 0.0001
```

Use `--data_dir /path/to/data` to select another data location, and `python Args.py --help` to inspect the options. The commands retain defaults for the remaining parameters. This snapshot selects checkpoints using test Recall@20, rather than the validation selection described in the paper; full paper results have not been revalidated.

If user/item pretrained embeddings are missing, generate them first:

```bash
python -X utf8 LightGCN.py --data office --cuda 0 --pretrain_epochs 100
```

This helper writes `uEmbeds_pt.npy` and `iEmbeds_pt.npy`. Regenerate auxiliary arrays after replacing them. It has not been established that this helper recreates the historical paper embeddings.

## License and attribution

Adapted from [InfoDCL](https://github.com/Qin-lab-code/InfoDCL), upstream commit `c747e980aa7d4740b40cd81dd0717cbd14ae2c6c`. [MIT](LICENSE) covers the AdaDCL authors' original contributions only. No upstream license was found in the inspected snapshot; redistribution permission for inherited code remains to be confirmed. Dataset terms are separate.
