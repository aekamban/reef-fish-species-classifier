# Reef Fish Species Classification
### Multimodal deep learning for fine-grained species identification using images, spatiotemporal metadata, and controlled augmentation experiments

I built this project to investigate how deep learning can improve fish species identification from real-world citizen-science images. The system uses approximately 5,600 iNaturalist observations across 16 tropical reef fish species and combines transfer learning, controlled data-scarcity experiments, taxonomic modeling, multimodal metadata fusion, and statistically tested augmentation strategies.

The project began as my graduate deep learning final project for DSP 566 at the University of Rhode Island and continued as an independent extension after the course submission. I wrote the modeling and data pipeline code for the original project and then extended it with multimodal fusion, redesigned augmentation, additional controls, and a stronger reproducibility workflow.

The strongest model combines an EfficientNetB0 image representation with latitude, longitude, and observation date, improving test accuracy from 78.33% image-only to 89.48% fused. The corresponding macro F1 improvement is +0.1098, with a 95% species-stratified paired-bootstrap CI of [+0.087, +0.132] and exact McNemar p < 0.0001.

> **Best model:** 89.48% test accuracy | 0.8948 macro F1  
> **Dataset:** 16 reef-fish species across 4 families | ~5.6K iNaturalist observations  
> **Stack:** Python, TensorFlow/Keras, EfficientNetB0, tf.data, NumPy, pandas, SciPy, scikit-learn, Google Colab

---

## Why this project matters

Citizen-science platforms collect wildlife imagery at a scale that cannot be reviewed manually by experts. Fish identification is a useful fine-grained classification problem because underwater images vary substantially in lighting, visibility, background, pose, and subject size, while closely related species can be visually similar.

Rather than treating the project as a single model-selection exercise, I used it to investigate five questions:

1. **Transfer learning:** How much does ImageNet pretraining help compared with random initialization?
2. **Data scarcity:** Can augmentation recover performance when real training observations are limited?
3. **Taxonomy:** Does explicitly modeling family structure improve species classification?
4. **Metadata fusion:** Does combining image features with geography and seasonality improve identification?
5. **Augmentation design:** Can a symmetric, underwater-motivated policy improve on the targeted augmentation strategy that failed in the first experiment?

The later experiments were motivated directly by limitations and unanswered questions uncovered in the earlier ones.

---

## Project design

### Dataset

I collected approximately 350 candidate observations per species from the iNaturalist API, using one photo per observation and retaining only reusable Creative Commons or public-domain licenses.

The final species set contains 16 species across 4 families, with 4 species per family. I selected the set deliberately so that:

- each class had enough observations for controlled data-scarcity experiments
- each family contained multiple species, making hierarchical family-to-species classification meaningful
- the images retained real citizen-science variability rather than being filtered to ideal benchmark photographs

The source images include clear close-ups as well as small, distant, partially obscured, low-contrast, and cluttered reef scenes.

### Frozen evaluation design

The project uses fixed per-species splits:

- **training:** ~250 observations per species
- **validation:** 40 per species
- **test:** 60 per species

For the controlled low-data experiment, eight species were reduced from roughly 250 to 50 training observations while validation and test remained unchanged.

This keeps the evaluation distribution fixed while changing only the training evidence available to the model.

---

## Modeling approach

### Transfer learning

I compared ResNet50 configurations trained from random initialization, initialized from ImageNet, frozen, and fine-tuned in stages. I also evaluated a frozen EfficientNetB0 backbone.

The strongest original image-only configuration was frozen EfficientNetB0 at approximately 79.3% test accuracy and 0.793 macro F1.

That became the backbone for the later multimodal experiments.

### Taxonomic modeling

I tested whether the model could benefit from the known biological hierarchy by comparing:

- flat 16-class species classification
- explicit family → species hierarchical probabilities
- auxiliary family supervision

The hard hierarchy produced essentially the same test performance as the flat ResNet50, showing that adding biologically meaningful structure does not automatically improve generalization.

---

## Multimodal image + metadata fusion

The data pipeline already retained latitude, longitude, and observation date for provenance. I extended the model to test whether those variables contained useful information beyond the image itself.

```text
224 x 224 image
      |
      v
EfficientNetB0
(frozen ImageNet backbone)
      |
      v
 image embedding --------\
                          \
                           --> concatenation --> dense head --> 16 species
                          /
lat / lon / date --------/
      |
      v
metadata encoder
```

The metadata representation is designed around the geometry of the variables:

- **longitude:** sinusoidal encoding because -180° and +180° are adjacent, not far apart
- **day of year:** sinusoidal encoding because December 31 and January 1 are adjacent
- **latitude:** linearly normalized because it does not wrap

I evaluated three ablations on the same held-out test set:

- image only
- metadata only
- image + metadata

This isolates the incremental value of each signal rather than attributing a gain to the fused model without a control.

---

## Augmentation redesign

The first augmentation experiment applied extra augmentation only to eight artificially data-scarce species. That policy reduced performance on the classes it was meant to help and also changed behavior on classes that were never augmented.

I used that failure to redesign the experiment around two hypotheses:

- applying stronger transforms to only some classes can create a class-specific train/test distribution mismatch
- underwater augmentation should reflect plausible image formation rather than generic image distortion

The revised policy is applied symmetrically across classes and uses underwater-motivated transforms such as mild color-cast variation, blur, and occlusion.

I compared:

- `no_aug`
- `targeted_original`
- `symmetric_new`

under the same imbalanced training condition.

---

## Results

Across the project, the experiments produced a mix of strong positive results, useful negative results, and one major improvement from multimodal modeling.

| Research question | Main result |
|---|---|
| **RQ1: Transfer learning** | Pretrained networks substantially outperformed random initialization; frozen EfficientNetB0 was the strongest image-only model at **79.27% accuracy / 0.7930 macro F1** |
| **RQ2: Data scarcity + targeted augmentation** | Reducing real training observations caused a large performance loss, and the original class-targeted augmentation policy made it worse |
| **RQ3: Taxonomic hierarchy** | Hard family → species hierarchy produced no meaningful improvement over flat classification |
| **RQ4: Metadata fusion** | Image + spatiotemporal metadata increased accuracy from 78.33% to 89.48% and macro F1 by +0.1098 |
| **RQ5: Augmentation redesign** | Symmetric underwater-motivated augmentation recovered +0.1132 macro F1 over the original targeted policy on low-data species |

---

### RQ1: How much does transfer learning help?

Pretrained visual representations were substantially more effective than training ResNet50 from random initialization.

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| ResNet50, random initialization | 19.58% | 0.1559 |
| ResNet50, ImageNet trainable | 61.77% | 0.6210 |
| ResNet50, ImageNet frozen | 69.27% | 0.6923 |
| ResNet50, staged fine-tuning | 72.92% | 0.7284 |
| **EfficientNetB0, ImageNet frozen** | **79.27%** | **0.7930** |

Within ResNet50, freezing the pretrained backbone outperformed making the full ImageNet backbone trainable immediately:

- +7.50 accuracy points, exact McNemar p < 0.0001

Staged fine-tuning improved further over frozen ResNet50:

- +3.65 accuracy points, p = 0.0014

Frozen EfficientNetB0 then improved another 10 points over frozen ResNet50:

- +10.0 accuracy points, p < 0.0001

The random-initialization comparison also strongly favored pretraining, with a paired-bootstrap macro F1 difference of +0.4651, 95% CI [+0.4289, +0.4999].

One caveat: the rescued random-init model required a different learning rate and training budget from the ImageNet-trainable model, so this establishes a large practical advantage for pretraining, but not a perfectly isolated initialization-only effect.

**Takeaway:** for this dataset size, pretrained visual features were critical, and preserving them initially before careful fine-tuning generalized better than immediately updating the full backbone.

---

### RQ2: Can targeted augmentation recover performance when real training data is limited?

To create a controlled data-scarcity problem, I reduced eight species from roughly 250 to 50 training observations while keeping validation and test distributions unchanged.

That manipulation had the expected effect:

- low-data species accuracy with full training data: 74.79%
- after thinning to 50 observations: 49.17%
- change: −25.63 percentage points, p < 0.0001

The original targeted augmentation strategy then made performance worse rather than recovering the loss:

- low-data accuracy: 49.17% → 36.04%
- additional change: −13.12 percentage points, p < 0.0001
- Δ macro F1: −0.1125
- 95% CI: [−0.1498, −0.0771]
- 7 of 8 low-data species lost per-class F1

The effect also propagated beyond the classes being augmented. Accuracy on the eight full-data species increased from 71.04% to 74.17%, while their macro F1 fell from 0.5985 to 0.5762.

That mixed result suggested the augmentation policy was changing the shared classifier's decision boundaries rather than simply adding useful variation to the low-data classes.

**Takeaway:** transformed versions of the same small pool of images did not substitute for genuinely different observations, and selectively changing only some classes created an undesirable distribution shift.

That failure directly motivated RQ5.

---

### RQ3: Does explicitly modeling taxonomic hierarchy improve species classification?

I compared flat 16-class ResNet50 classification with a model that explicitly factors species probabilities through fish family.

The result was essentially null.

| Model | Test accuracy |
|---|---:|
| Flat ResNet50 | 69.27% |
| Hierarchical ResNet50 | 69.17% |

The paired comparison found:

- accuracy difference: −0.10 percentage points
- exact McNemar p = 1.0000
- Δ macro F1: −0.0016
- 95% CI: [−0.0228, +0.0209]

Importantly, the hierarchy had room to help. The flat model inferred the correct family for only 82.6% of test images, and 167 species errors crossed family boundaries. The hierarchical model nevertheless produced the same 82.6% family accuracy when family was inferred from its species prediction.

The size of the null result was also small relative to ordinary training variation. Across three frozen ResNet50 runs with different random seeds:

- macro F1 SD: 0.0071
- macro F1 range: 0.0128

The hierarchy difference of 0.0016 was therefore much smaller than the variation observed from retraining the model.

An auxiliary multitask model that predicted both family and species reached 0.7593 validation macro F1, suggesting that family supervision may still be useful even when a hard hierarchical probability constraint is not. That result was validation-only and was not used as a final test-set claim.

**Takeaway:** biologically meaningful structure did not automatically improve classification. The hard hierarchy added complexity without extracting information the flat classifier had missed.

---

### RQ4: Does metadata improve image classification?

The original image pipeline retained latitude, longitude, and observation date for provenance but did not use them as model inputs. I extended the model with a metadata branch and evaluated image-only, metadata-only, and fused models on the same held-out test set.

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| Best image-only result from the first modeling stage | 79.27% | 0.7930 |
| v2 image-only reproduction | 78.33% | 0.7850 |
| Metadata-only | 34.06% | 0.3189 |
| **Image + metadata fusion** | **89.48%** | **0.8948** |

The v2 image-only rerun landed close to the earlier image-only result, providing a useful reproduction check before interpreting the extension.

| Comparison | Δ accuracy | Δ macro F1, 95% CI | McNemar p |
|---|---:|---:|---:|
| **Fused vs image-only** | **+11.15 pts** | **+0.1098 [+0.087, +0.132]** | **< 0.0001** |
| Metadata-only vs image-only | −44.27 pts | −0.4661 [−0.501, −0.429] | < 0.0001 |
| Fused vs metadata-only | +55.42 pts | +0.5759 [+0.540, +0.609] | < 0.0001 |

Metadata alone is not competitive with image classification, but it clearly contains complementary information. Combining the two signals produced the strongest model in the project.

**Takeaway:** visual appearance and spatiotemporal context provide different information, and multimodal fusion substantially improved classification over either input source alone.

This result also creates an important deployment question. Geographic metadata can become a shortcut if the training and deployment distributions differ, so a production system should be tested under geographic holdouts before assuming the full gain will generalize.

---

### RQ5: Can a better augmentation policy recover low-data performance?

The RQ2 failure suggested two problems with the original augmentation design:

1. stronger transformations were applied only to the low-data classes, creating a class-specific training distribution
2. the transformations were not specifically designed around real underwater image conditions

I redesigned the policy to apply augmentation **symmetrically across classes** and use underwater-motivated changes such as mild color-cast variation, blur, and occlusion.

Using the same imbalanced training condition:

| Policy | Macro F1, all species | Macro F1, low-data | Macro F1, full-data |
|---|---:|---:|---:|
| `no_aug` | 0.7010 | 0.7281 | 0.6739 |
| `targeted_original` | 0.6361 | 0.6369 | 0.6353 |
| **`symmetric_new`** | **0.7124** | **0.7454** | **0.6794** |

For the eight low-data species:

`symmetric_new` vs `targeted_original`: Δ macro F1 = +0.1132, 95% CI [+0.079, +0.150].

The redesigned policy therefore significantly recovered performance relative to the original targeted strategy.

The broader claim is intentionally narrower:

- `symmetric_new` significantly outperformed `targeted_original` on the low-data classes
- the full-data difference between those policies was not statistically significant
- `symmetric_new` was not directly tested against `no_aug` using the same paired-bootstrap comparison, so the current experiment does **not** establish that extra augmentation is better than no extra augmentation

**Takeaway:** redesigning the augmentation policy fixed a clear failure mode of the original experiment, but the evidence supports improvement over the targeted policy, not yet superiority over doing no additional augmentation.

---

## Overall finding

The progression across the five experiments mattered as much as the final model.

The project began by showing that pretrained visual representations were far more effective than training from scratch. It then exposed two approaches that did not behave as expected: class-targeted augmentation hurt low-data performance, and hard taxonomic hierarchy added essentially no predictive value.

Those negative results generated the next hypotheses rather than ending the analysis. Redesigning augmentation recovered the performance lost under the original policy, while adding spatiotemporal context produced the largest improvement in the project: 89.48% test accuracy and 0.8948 macro F1.

The strongest lesson is therefore not simply that one architecture won. It is that controlled experiments, failure analysis, and multimodal feature design were more valuable than adding model complexity without evidence.

---

## Experimental rigor

I designed the evaluation around controlled comparisons rather than headline accuracy alone:

- fixed train / validation / test splits
- macro F1 as the primary class-balanced metric
- exact McNemar tests for paired accuracy comparisons
- species-stratified paired bootstrap, 2,000 resamples, for macro F1 differences
- image-only, metadata-only, and fused ablation controls
- controlled reduction of real training observations for the data-scarcity experiment
- seed-sensitivity analysis in the original modeling stage
- synthetic-data tests to validate multimodal model wiring independently of predictive performance
- explicit reporting of negative and null results

The goal was to determine not only whether a model scored higher, but which change caused the improvement, how large the effect was, and what the experiment could legitimately support.

---

## Result summary

| Run | Accuracy | Macro F1 | Low-data F1 | Full-data F1 | Worst class |
|---|---:|---:|---:|---:|---|
| **v2_D_fused** | **0.8948** | **0.8948** | **0.9015** | **0.8881** | *Acanthurus tractus* |
| v2_D_image_only | 0.7833 | 0.7850 | 0.8302 | 0.7399 | *Abudefduf saxatilis* |
| v2_E_symmetric_new | 0.7104 | 0.7124 | 0.7454 | 0.6794 | *Abudefduf saxatilis* |
| v2_E_no_aug | 0.7042 | 0.7010 | 0.7281 | 0.6739 | *Abudefduf saxatilis* |
| v2_E_targeted_original | 0.6344 | 0.6361 | 0.6369 | 0.6353 | *Acanthurus tractus* |
| v2_D_metadata_only | 0.3406 | 0.3189 | 0.3023 | 0.3355 | *Naso unicornis* |

---

## ML engineering and data science decisions

This project demonstrates:

- **Transfer learning:** ResNet50 and EfficientNetB0 with ImageNet representations
- **Multimodal modeling:** fusion of visual and spatiotemporal features
- **Feature engineering:** cyclic encoding for periodic variables
- **TensorFlow pipelines:** `tf.data` loading, preprocessing, batching, and augmentation
- **Controlled experimentation:** ablations, fixed evaluation sets, and data-scarcity manipulation
- **Statistical model comparison:** paired bootstrap confidence intervals and McNemar tests
- **Failure analysis:** using an unsuccessful augmentation experiment to formulate and test a better hypothesis
- **Reproducibility:** shared libraries, fixed splits, executed notebook outputs, synthetic sanity checks, and cached data staging
- **Data governance:** Creative Commons license handling and explicit limits on dataset redistribution

---

## Data and ethics

The source data are iNaturalist "research grade" citizen-science observations. I intentionally retained real-world variation such as small subjects, cluttered reef backgrounds, variable lighting, and partial occlusion rather than filtering the dataset down to ideal photographs.

The repository does not redistribute the image dataset.

Images were collected under CC0, CC-BY, CC-BY-NC, CC-BY-SA, or CC-BY-NC-SA licenses, and most images in the dataset were non-commercial CC-BY-NC. Metadata can also include contributor information and precise wildlife locations.

For those reasons, this repository contains code, methodology, and experiment outputs rather than republishing source images or precise location data.

For a deployed ecological system, I would additionally:

- obscure locations for rare or sensitive species
- evaluate performance under geographic distribution shift
- expose confidence and top-k predictions rather than presenting model outputs as authoritative labels
- retain human review for uncertain or high-impact identifications

---

## Repository structure

```text
reef-fish-species-classifier/
├── fishlib.py               # core data/model/evaluation library
├── fishlib_v2.py            # multimodal fusion + redesigned augmentation
├── train_v2.ipynb           # executed Colab extension notebook; outputs included
├── test_v2_synthetic.py     # synthetic sanity checks for v2 model/data wiring
├── requirements.txt
└── LICENSE
```

`data/images/` and `splits.csv` are intentionally not included.

---

## Reproducing the experiments

1. Install the dependencies in `requirements.txt`.
2. Recreate or provide the iNaturalist dataset and frozen `splits.csv`.
3. Open `train_v2.ipynb` in Google Colab with a GPU runtime.
4. Set `PROJECT_DIR` to the project directory in Google Drive.
5. Run the data-staging cells.
6. Run the training and evaluation cells in order.

The notebook caches the staged dataset to reduce repeated Drive I/O on future runs.

### Current artifact limitation

The completed Colab extension produced the metrics reported above, and those outputs are preserved in the executed notebook. The runtime disconnected before the trained `.keras` checkpoints and full per-class prediction JSONs were copied to persistent storage.

Because of that, reproducing confusion matrices or deeper per-species diagnostics requires rerunning the affected experiments. A future run should persist `results_v2/` to Drive immediately after each experiment rather than only at notebook completion.

I include this limitation explicitly because reproducibility includes documenting what artifacts were and were not preserved.

---

## Project context

I developed this project for DSP 566: Advanced Topics in Machine Learning in the University of Rhode Island's M.S. in Data Science program and continued extending it independently after the course project was completed. I found the dataset, proposed the topic, and conducted a literature review.

I authored the project's data collection, preprocessing, modeling, evaluation, and extension code. For the original course submission, classmates Mario Corado and Dylan Goldrick helped run experiments and collaborated with me on the written report and presentation.

This repository presents the technical project as a single evolving body of work while acknowledging that course deliverables were completed collaboratively.

---

## Next experiments

The highest-value next steps are:

1. **Geographic generalization:** evaluate the fused model using region-blocked or leave-one-geography-out splits to measure how much of the metadata gain survives distribution shift.
2. **Metadata contribution analysis:** rerun with persisted predictions and measure which species gain most from geography and seasonality.
3. **Augmentation control:** directly test `symmetric_new` vs `no_aug` using the same paired bootstrap procedure.
4. **Subject-aware preprocessing:** compare full-scene images with fish-focused crops or higher-resolution inputs.
5. **Production persistence:** automatically save checkpoints, predictions, metrics, and run metadata to durable storage after every experiment.

---

## License

Code in this repository is released under the MIT License.

The iNaturalist images are not covered by the repository's MIT license. Each source image remains subject to its own Creative Commons or public-domain license and attribution requirements.
