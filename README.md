# Reef Fish Species Classifier v2: Metadata Fusion + Revised Augmentation

**A solo extension of a course project's deep-learning fish classifier, adding geo/temporal metadata fusion and a redesigned augmentation policy on top of the original transfer-learning baseline. Both hypotheses were tested with a real GPU training run; results below are real numbers, not projections.**

## Attribution

The 16-species classifier this repo builds on was a 3-person team project for a course at the University of Rhode Island (DSP 566), with Dylan Goldrick and Mario Corado. Within that team project:

- **The data pipeline and shared library, `build_dataset.py`, `make_splits.py`, `species_recon.py`, and `fishlib.py` (reproduced here unchanged), were authored solely by Abi Kambanis**, per file ownership in the team's shared Drive.
- **The modeling experiments (the ResNet50/EfficientNetB0 transfer-learning runs, the targeted-augmentation and hierarchical-classification arms), the results, and the written report were produced collaboratively by the team.** That baseline result (frozen EfficientNetB0, 79.3% test accuracy/macro F1) is a team result and is not this repo's contribution.

**Everything in this repo beyond the reproduced `fishlib.py` (`fishlib_v2.py`, `train_v2.ipynb`, and the metadata-fusion / augmentation-redesign work they implement) is solo follow-up work**, built independently after the course ended, extending the team's baseline with two things it explicitly left unresolved.

## What this tests

The original project's own report named two open questions this repo answers:

- **RQ4, metadata fusion.** The original collected latitude/longitude and observation date for every image but excluded them from every model ("all modeling in this project is based on image content alone"). Several of the 16 species split cleanly by ocean basin (e.g. *Thalassoma pavo* is Eastern Atlantic/Mediterranean, *Naso unicornis* is Indo-Pacific), so geography isn't a weak signal here. `fishlib_v2.py` adds a small metadata sub-network, sinusoidal encoding of longitude and day-of-year (both of which wrap, so raw scalars would put adjacent values maximally far apart) plus linear-normalized latitude (which doesn't wrap), fused with the image embedding before the dense head. An image-only / metadata-only / fused ablation, scored with the same paired-bootstrap and McNemar tests the original used, isolates exactly how much of any gain is geography doing work the original never tested.

- **RQ5, augmentation redesign.** The original's targeted augmentation (extra contrast/brightness applied only to 8 artificially data-scarce species) backfired: it dropped those species' F1 further and measurably shifted performance on the untouched classes too. The report's own diagnosis was that class-targeted augmentation shifted the low-data classes' training distribution away from the untouched validation/test distribution. `fishlib_v2.py`'s `realistic_augment()` tests that diagnosis directly: augmentation applied symmetrically across all classes (no targeting), using transforms with a physical reason to appear in underwater citizen-science photos, color-cast jitter, mild blur, occlusion, rather than generic contrast/brightness.

## Results

Trained on Colab (GPU runtime, EfficientNetB0 backbone, frozen, matching the original's best-performing configuration) against the same 16-species dataset and splits as the original project. All comparisons use the original's own statistical machinery: exact McNemar tests for accuracy, and species-stratified paired bootstrap (2000 resamples) for macro F1, so a result only counts as real here if its 95% CI excludes zero.

### RQ4: metadata fusion is a clear, significant win

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Original team baseline (image-only, reported) | 79.3% | 79.3% |
| **v2 image-only** (this repo, same protocol) | 78.33% | 0.7850 |
| **v2 metadata-only** (lat/long/date alone) | 34.06% | 0.3189 |
| **v2 fused** (image + metadata) | **89.48%** | **0.8948** |

The image-only rerun (78.33%) lands within normal run-to-run variance of the original team's reported 79.3%, a useful sanity check that this reproduction is faithful before trusting the comparison built on top of it.

| Comparison | Δ accuracy | Δ macro F1 (95% CI) | McNemar p |
|---|---|---|---|
| fused vs. image-only | **+11.15 pts** | **+0.1098** [+0.087, +0.132] | <0.0001 |
| metadata-only vs. image-only | −44.27 pts | −0.4661 [−0.501, −0.429] | <0.0001 |
| fused vs. metadata-only | +55.42 pts | +0.5759 [+0.540, +0.609] | <0.0001 |

Metadata alone is a weak classifier (34% on 16 classes, but still ~5x random chance, so geography does carry real signal on its own), and image alone is strong. Fusing the two is not just "as good as image alone", it's a significant, sizable improvement over either signal in isolation: **+11 accuracy points, 95% CI clear of zero, p < 0.0001.** This is the headline result: the original report's untested hypothesis, that the excluded metadata was leaving real signal on the table, holds up under a real, controlled test.

### RQ5: augmentation redesign is a real but narrower win

Same imbalanced-condition setup as the original's Arm B (8 species thinned to 50 training images), all three runs on identical frozen EfficientNetB0:

| Policy | Macro F1 (all 16) | Macro F1 (8 low-data) | Macro F1 (8 full-data) |
|---|---|---|---|
| `no_aug` (control, base flip/rotate/zoom only) | 0.7010 | 0.7281 | 0.6739 |
| `targeted_original` (reproduces the original's Arm B policy) | 0.6361 | 0.6369 | 0.6353 |
| **`symmetric_new`** (this repo's policy) | **0.7124** | **0.7454** | 0.6794 |

This replicates the original finding first: `targeted_original` is the worst of the three, confirming that class-targeted augmentation actively hurts rather than helps. `symmetric_new` beats it, but here's the honest, precise version of what's significant and what isn't:

- **`symmetric_new` vs. `targeted_original`, low-data classes** (the population targeted augmentation was supposed to help): Δ macro F1 = **+0.1132**, 95% CI **[+0.079, +0.150]**, excludes zero. Real, significant win.
- **`symmetric_new` vs. `targeted_original`, full-data classes** (checking whether the spillover onto untouched classes is fixed): Δ macro F1 = −0.0033, 95% CI **[−0.036, +0.030]**, includes zero. Not statistically significant. The raw numbers point the right direction (`targeted_original`'s full-data score of 0.6353 sits below the `no_aug` control's 0.6739, consistent with spillover; `symmetric_new`'s 0.6794 looks back to normal), but that specific comparison wasn't tested with the rigor the low-data one was, so "spillover is fixed" is a suggestive pattern here, not a tested claim.
- **`symmetric_new` vs. `no_aug`** was not directly tested. The raw margin is small (+0.0114 macro F1 overall), so "symmetric augmentation beats doing nothing" isn't something this run established either way.

So the precise claim: **symmetric augmentation fixes what was broken about the original's targeted approach, specifically and significantly on the classes it targeted.** Whether it's meaningfully better than no extra augmentation at all, or whether it truly eliminates cross-class spillover rather than just not showing it at this sample size, is still open.

### Full summary table

| run | accuracy | macro_f1 | macro_f1_low | macro_f1_full | worst class |
|---|---|---|---|---|---|
| v2_D_fused | 0.8948 | 0.8948 | 0.9015 | 0.8881 | *Acanthurus tractus* |
| v2_D_image_only | 0.7833 | 0.7850 | 0.8302 | 0.7399 | *Abudefduf saxatilis* |
| v2_E_symmetric_new | 0.7104 | 0.7124 | 0.7454 | 0.6794 | *Abudefduf saxatilis* |
| v2_E_no_aug | 0.7042 | 0.7010 | 0.7281 | 0.6739 | *Abudefduf saxatilis* |
| v2_E_targeted_original | 0.6344 | 0.6361 | 0.6369 | 0.6353 | *Acanthurus tractus* |
| v2_D_metadata_only | 0.3406 | 0.3189 | 0.3023 | 0.3355 | *Naso unicornis* |

## What's not here, and why

The Colab runtime disconnected after training completed but before the `.keras` checkpoints and full per-class eval JSONs (confusion matrices, raw predictions) were copied out of the ephemeral session to persistent storage. The numbers above are the complete, real output captured in the notebook's saved cell outputs (`train_v2.ipynb`, included in this repo exactly as executed) before that happened, they aren't reconstructed or estimated. What's lost is the ability to go beyond those numbers: a confusion matrix, a full 16-class per-species F1 breakdown, or checking whether the metadata gain concentrates in geographically-separated species pairs, without rerunning.

`train_v2.ipynb`'s data-staging cells now save a local `data_cache.zip` back to Drive so a rerun doesn't re-pay the slow first-load cost; a results-saving step (copying `results_v2/` back to Drive right after training, not just at the end of the notebook) is the natural next addition before any future run.

## Repository structure

```
reef-fish-species-classifier/
├── fishlib.py              Original team project's shared library, unmodified (see Attribution)
├── fishlib_v2.py            Metadata encoding, fusion model, revised augmentation
├── train_v2.ipynb           Colab notebook, executed, outputs included, real run behind the Results above
├── test_v2_synthetic.py     Synthetic-data sanity checks for fishlib_v2.py (validates wiring, not accuracy)
├── requirements.txt
└── LICENSE
```

`data/images/` and `splits.csv` are not included, see Data below.

## Data

Images are citizen-science observations pulled from the iNaturalist API under CC0/CC-BY/CC-BY-NC/CC-BY-SA/CC-BY-NC-SA licenses. Per the original project's data ethics review, full redistribution of the source images isn't appropriate given the mixed license terms (~86% are CC-BY-NC, non-commercial only) and potential contributor-privacy considerations around retained metadata, so this repo ships code, methodology, and results, not the dataset itself. To reproduce: use the original `build_dataset.py`/`make_splits.py` pipeline (or your own saved `data/images/` + `splits.csv` from the original project) to regenerate a locally-held copy.

## Reproducing this

1. Open `train_v2.ipynb` in Colab (GPU runtime).
2. Run the Drive-mount cell and set `PROJECT_DIR` to wherever your copy of the project (containing `data/images/`, `splits.csv`, `fishlib.py`, `fishlib_v2.py`) lives in Drive.
3. Run the data-staging cell. First run zips `data/images/` + `splits.csv` and caches the zip back to Drive (slow, one time); every run after that just copies the cached zip (fast).
4. Run the rest of the notebook. Results (eval JSONs, checkpoints) are written to `results_v2/` on local Colab disk, copy that folder back to Drive (or download it) before the runtime disconnects, it is not saved automatically.

## License

MIT (code only, see Data above for image licensing).
