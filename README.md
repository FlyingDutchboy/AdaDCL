# AdaDCL: Adaptive Noise Enhanced Diffusion Contrastive Learning for Recommendation

PyTorch implementation of our paper **"AdaDCL: Adaptive Noise Enhanced Diffusion Contrastive Learning for Recommendation"**, accepted by **Neurocomputing**.

Paper: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231226020424) | [DOI](https://doi.org/10.1016/j.neucom.2026.134644)

AdaDCL addresses semantic drift in sparse recommendation by learning structured noise from latent preference distributions. A Variational Autoencoder (VAE) captures the underlying structure, while FiLM introduces auxiliary semantics.

The resulting signals guide diffusion-based view generation. A joint training objective combines preference ranking, contrastive learning, reconstruction, and regularization.

**Authors:** Renjie Tian, Minxi Yang, Feng Xie, Dahua Gao, and Zhenyuan Lin.

**Affiliation:** School of Artificial Intelligence, Xidian University.

## Architecture

AdaDCL contains three main components:

1. **Structured Noise Generation:** A FiLM-aided VAE learns data-dependent noise, conditioned on auxiliary information derived from user-item interactions. A context encoder further transforms the generated noise.
2. **Diffusion-based Contrastive Learning:** The diffusion model reconstructs item representations to form augmented views for contrastive learning.
3. **Joint Optimization:** BPR ranking, contrastive, reconstruction, KL, and regularization terms are optimized together to learn recommendation representations.

The noise-generation and diffusion modules are implemented in [Modules_ori.py](Modules_ori.py) and [diff.py](diff.py). The training procedure is in [Coach.py](Coach.py).

## Requirements

The paper reports **Python 3.11.14**, **PyTorch 2.2.2**, and **CUDA 12.1**. The runtime dependencies are:

```text
torch>=2.2.2,<3
numpy>=1.24.4,<3
scipy>=1.10.1,<2
matplotlib>=3.8,<4
tqdm>=4.66,<5
```

Install a PyTorch build compatible with your hardware, then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

The experiments use **Amazon-Office**, **MovieLens-1M**, and **Yelp**. Original benchmark sources:

- [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/)
- [MovieLens](https://grouplens.org/datasets/movielens/1m/)
- [Yelp](https://www.yelp.com/dataset)

Processed splits and pretrained embeddings are supplied separately. Raw benchmark downloads do not recreate the processed experiment inputs automatically. Place matching files under `data/`:

```text
data/
  office/                   # or ml-1m/ or yelp/
    train.txt
    test.txt
    uEmbeds_pt.npy
    iEmbeds_pt.npy
    item_social_emb.npy
    item_knowledge_emb.npy
```

Each row of `train.txt` and `test.txt` contains `user_id item_id`. IDs start at zero and must match embedding row indices.

User/item embeddings have shape `[n_users, 64]` and `[n_items, 64]`; auxiliary arrays have shape `[n_items, 64]`.

The two auxiliary item arrays and graph cache are generated if missing. Auxiliary generation can be expensive on large datasets. Rebuild caches when changing the split or pretrained embeddings.

### 2. Training

Run the following commands from the directory containing `main.py`:

```bash
# Amazon-Office
python -X utf8 main.py --data office --lr 0.005 --batch_size 2048 --timesteps 100 --beta 0.001

# MovieLens-1M
python -X utf8 main.py --data ml-1m --lr 0.005 --batch_size 8192 --timesteps 200 --bpr_alpha 0.9 --recon_weight 0.1 --beta 0.0001

# Yelp
python -X utf8 main.py --data yelp --lr 0.01 --batch_size 8192 --timesteps 200 --bpr_alpha 0.9 --recon_weight 0.1 --beta 0.0001
```

Use `--cuda 0` to select GPU 0 or `--data_dir /path/to/data` for another data location. The code uses the CPU when CUDA is unavailable. Logs and curves are saved in `log/`, and checkpoints in `result/`.

See `python Args.py --help` for all options. These commands retain the defaults for other parameters.

If user/item pretrained embeddings are missing, generate them before training:

```bash
python -X utf8 LightGCN.py --data office --cuda 0 --pretrain_epochs 100
```

Replace `office` with the desired dataset. This helper writes `uEmbeds_pt.npy` and `iEmbeds_pt.npy`; regenerate auxiliary arrays after replacing them. Its outputs have not been verified against the historical paper embeddings.

**Implementation note:** This snapshot does not read `valid.txt` and selects checkpoints using test Recall@20, whereas the paper describes validation-based selection. Full paper results have not been revalidated for this package.

## Implemented Models

| Model | Paper | Type | Code |
| --- | --- | --- | --- |
| AdaDCL | Tian et al., [AdaDCL: Adaptive Noise Enhanced Diffusion Contrastive Learning for Recommendation](https://doi.org/10.1016/j.neucom.2026.134644), Neurocomputing, 2026 | Contrastive Learning + Diffusion | [PyTorch](main.py) |

## Related Datasets

The paper evaluates AdaDCL on three benchmark datasets. The reported dataset statistics are:

| Datasets | Amazon-Office | ML-1M | Yelp |
| --- | ---: | ---: | ---: |
| #Users | 4,905 | 6,040 | 29,601 |
| #Items | 2,420 | 3,706 | 24,734 |
| #Interactions | 53,258 | 1,000,209 | 1,069,128 |
| Sparsity | 99.55% | 95.53% | 99.85% |

## Reference

Please cite the paper when using AdaDCL in your research:

```bibtex
@article{tian2026adadcl,
  title   = {{AdaDCL}: Adaptive noise enhanced diffusion contrastive learning for recommendation},
  author  = {Tian, Renjie and Yang, Minxi and Xie, Feng and Gao, Dahua and Lin, Zhenyuan},
  journal = {Neurocomputing},
  year    = {2026},
  doi     = {10.1016/j.neucom.2026.134644}
}
```

## Acknowledgments and License

This implementation builds on [InfoDCL](https://github.com/Qin-lab-code/InfoDCL). We thank its authors for making the code publicly available.

The [MIT license](LICENSE) covers original AdaDCL contributions only. 
