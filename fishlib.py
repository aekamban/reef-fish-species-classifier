"""
Original DSP 566 course project shared library (data pipeline, model zoo,
training/eval/statistics helpers), unmodified except for this header.

Authored solely by Abi Kambanis (per file ownership in the team's shared
Drive), as part of a 3-person team project (Dylan Goldrick, Mario Corado,
Abi Kambanis) for a course at the University of Rhode Island -- the
modeling experiments, results, and written report built on top of this
library were produced collaboratively by the team. Reproduced here
unchanged because fishlib_v2.py imports and extends it directly -- see
README.md for the full attribution breakdown.
"""
import csv, collections, json, os, glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

__version__ = '3.1'      # bump when this file changes; the import cell asserts on it

SEED = 42
IMG_SIZE = (224, 224)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE
SPLITS = 'splits.csv'
RESULTS = 'results'

# The eight species whose TRAINING pool is thinned for the augmentation arm.
# Validation and test stay balanced at 40 and 60 for every species.
LOW_DATA = {
    'Halichoeres bivittatus', 'Labroides dimidiatus',
    'Chromis chromis', 'Amphiprion clarkii',
    'Acanthurus tractus', 'Naso unicornis',
    'Chaetodon capistratus', 'Chaetodon vagabundus',
}


# ---------------------------------------------------------------- data
def _rows(condition):
    rows = list(csv.DictReader(open(SPLITS)))
    if condition == 'imbalanced':
        rows = [r for r in rows
                if not (r['split'] == 'train' and r['train_low_data'] == 'drop')]
    elif condition != 'balanced':
        raise ValueError("condition must be 'balanced' or 'imbalanced'")
    return rows


def class_names():
    return sorted({r['scientific_name'] for r in csv.DictReader(open(SPLITS))})


def family_names():
    return sorted({r['family'] for r in csv.DictReader(open(SPLITS))})


def family_map():
    """species index -> family index."""
    rows = list(csv.DictReader(open(SPLITS)))
    sp, fam = class_names(), family_names()
    lookup = {r['scientific_name']: r['family'] for r in rows}
    return {i: fam.index(lookup[s]) for i, s in enumerate(sp)}, fam


def genus_map():
    """species index -> genus name. Within-family is not the same as
    within-genus, and the distinction matters for fine-grained error analysis."""
    rows = list(csv.DictReader(open(SPLITS)))
    lookup = {r['scientific_name']: r['genus'] for r in rows}
    return {i: lookup[s] for i, s in enumerate(class_names())}


def membership_matrix():
    """M[s, f] = 1 if species s belongs to family f."""
    fmap, fams = family_map()
    M = np.zeros((len(fmap), len(fams)), dtype='float32')
    for s, f in fmap.items():
        M[s, f] = 1.0
    return M


def low_data_indices():
    return [i for i, n in enumerate(class_names()) if n in LOW_DATA]


def _ds(rows, training, augment=None, targeted=False, hierarchical=False):
    """targeted=True augments ONLY images whose label is in LOW_DATA.

    Augmentation vs. targeted augmentation for classes that are short of data",
    which is the question Ben Tamou et al. ask and the one our research question states."""
    names = class_names()
    idx = {n: i for i, n in enumerate(names)}
    fmap, _ = family_map()
    paths = [r['file_path'] for r in rows]
    ys = [idx[r['scientific_name']] for r in rows]

    def load(p, y):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.uint8), y     # uint8 keeps the RAM cache small

    if hierarchical:
        fs = [fmap[y] for y in ys]
        d = tf.data.Dataset.from_tensor_slices((paths, list(zip(ys, fs))))

        def load_h(p, yf):
            img, _ = load(p, 0)
            return img, yf
        d = d.map(load_h, num_parallel_calls=AUTOTUNE)
    else:
        d = tf.data.Dataset.from_tensor_slices((paths, ys))
        d = d.map(load, num_parallel_calls=AUTOTUNE)

    d = d.cache()                            # decode each JPEG once per session
    if training:
        d = d.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    d = d.batch(BATCH)
    d = d.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)

    if augment is not None:
        low = tf.constant(low_data_indices(), dtype=tf.int32)

        def apply_aug(x, y):
            xa = augment(x, training=True)
            if not targeted:
                return xa, y
            sp = tf.cast(y[:, 0] if hierarchical else y, tf.int32)
            is_low = tf.reduce_any(tf.equal(tf.expand_dims(sp, 1),
                                            tf.expand_dims(low, 0)), axis=1)
            return tf.where(tf.reshape(is_low, [-1, 1, 1, 1]), xa, x), y
        d = d.map(apply_aug, num_parallel_calls=AUTOTUNE)

    if hierarchical:
        d = d.map(lambda x, y: (x, {'species': y[:, 0], 'family': y[:, 1]}),
                  num_parallel_calls=AUTOTUNE)
    return d.prefetch(AUTOTUNE)


def load_splits(condition='balanced', augment=None, targeted=False,
                hierarchical=False):
    rows = _rows(condition)
    g = collections.defaultdict(list)
    for r in rows:
        g[r['split']].append(r)
    return (_ds(g['train'], True, augment, targeted, hierarchical),
            _ds(g['val'], False, None, False, hierarchical),
            _ds(g['test'], False, None, False, hierarchical))


# ---------------------------------------------------------------- models
BACKBONES = {
    'resnet50':       (keras.applications.ResNet50,
                       keras.applications.resnet50.preprocess_input),
    'efficientnetb0': (keras.applications.EfficientNetB0,
                       keras.applications.efficientnet.preprocess_input),
    'mobilenetv2':    (keras.applications.MobileNetV2,
                       keras.applications.mobilenet_v2.preprocess_input),
}


def default_augment(extra=False):
    L = [layers.RandomFlip('horizontal'),
         layers.RandomRotation(0.08),
         layers.RandomZoom(0.15)]
    if extra:
        L += [layers.RandomContrast(0.2), layers.RandomBrightness(0.2)]
    return keras.Sequential(L, name='augment')

try:
    register_keras_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_keras_serializable = keras.utils.register_keras_serializable  # pre-Keras-3 location


@register_keras_serializable(package='fishlib')
class HierarchicalSoftmax(layers.Layer):
    """P(species) = P(family of species) * P(species | that family).

    The conditional softmax is taken within each family only, then scaled by that
    family probability. A species can therefore never receive more probability
    mass than its own family, which is what makes this a hierarchy rather than
    two heads that share a trunk and exchange hints."""

    def __init__(self, membership, **kw):
        super().__init__(**kw)
        self.membership = np.asarray(membership, dtype='float32')

    def build(self, input_shape):
        self.M = tf.constant(self.membership)          # [n_species, n_family]
        super().build(input_shape)

    def call(self, inputs):
        logits, fam_p = inputs                          # [B, n_sp], [B, n_fam]
        e = tf.exp(logits - tf.reduce_max(logits, axis=1, keepdims=True))
        denom_fam = tf.matmul(e, self.M)                # sum of exp within family
        denom_sp = tf.matmul(denom_fam, self.M, transpose_b=True)
        cond = e / (denom_sp + 1e-9)                    # P(species | its family)
        fam_p_sp = tf.matmul(fam_p, self.M, transpose_b=True)
        return cond * fam_p_sp

    def get_config(self):
        c = super().get_config()
        c['membership'] = self.membership.tolist()
        return c


def _trunk(inp, backbone, mode):
    """mode:
        'scratch'            small CNN, no pretraining
        'random_init'        backbone, weights=None, fully trainable
        'imagenet_trainable' backbone, ImageNet weights, fully trainable
        'frozen'             backbone, ImageNet weights, frozen

    random_init and imagenet_trainable are the matched pair for RQ1: identical
    architecture, trainability, learning rate and epoch budget, differing only in
    whether the backbone starts from ImageNet weights. Neither passes an explicit
    training flag to the base, so BatchNorm behaves identically in both."""
    if mode == 'scratch':
        x = layers.Rescaling(1. / 255)(inp)
        for f in (32, 64, 128, 256):
            x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
            x = layers.MaxPooling2D(2)(x)
        return layers.GlobalAveragePooling2D()(x)

    cls, prep = BACKBONES[backbone]
    if mode in ('random_init', 'imagenet_trainable'):
        w = None if mode == 'random_init' else 'imagenet'
        base = cls(include_top=False, weights=w, input_shape=(*IMG_SIZE, 3))
        base.trainable = True
        x = base(prep(inp))          # no explicit training flag: matched protocol
        return layers.GlobalAveragePooling2D()(x)

    base = cls(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    # training=False keeps BatchNorm in inference mode on the ImageNet moving
    # statistics. Chollet Ch. 8: letting BN update during fine-tuning undoes the
    # pretrained representation.
    x = base(prep(inp), training=False)
    return layers.GlobalAveragePooling2D()(x)


def build_model(backbone='resnet50', mode='frozen', n_classes=16,
                dense=256, dropout=0.4, seed=SEED):
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(*IMG_SIZE, 3))
    x = _trunk(inp, backbone, mode)
    x = layers.Dense(dense, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inp, out)


def build_hierarchical_model(backbone='resnet50', mode='frozen',
                             dense=256, dropout=0.4, seed=SEED):
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(*IMG_SIZE, 3))
    x = _trunk(inp, backbone, mode)
    x = layers.Dense(dense, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    fam = layers.Dense(len(family_names()), activation='softmax', name='family')(x)
    sp_logits = layers.Dense(len(class_names()), name='species_logits')(x)
    sp = HierarchicalSoftmax(membership_matrix(), name='species')([sp_logits, fam])
    return keras.Model(inp, {'species': sp, 'family': fam})


def build_multitask_model(backbone='resnet50', mode='frozen',
                          dense=256, dropout=0.4, seed=SEED):
    """Intermediate control for Arm C: auxiliary family supervision, but no
    probability constraint linking the heads.

    The hierarchical model differs from the flat model: it
    adds a family loss, and it constrains species probabilities by family. This
    model has the first without the second, so the three together separate the
    two mechanisms."""
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(*IMG_SIZE, 3))
    x = _trunk(inp, backbone, mode)
    x = layers.Dense(dense, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    fam = layers.Dense(len(family_names()), activation='softmax', name='family')(x)
    sp = layers.Dense(len(class_names()), activation='softmax', name='species')(x)
    return keras.Model(inp, {'species': sp, 'family': fam})


def unfreeze_top(model, n_layers=30):
    """Staged fine-tuning: call this on an already-trained frozen model.

    While the head is still random its large
    gradients flow straight into the pretrained filters and damage them, which is
    the standard warning in Chollet Ch. 8. BatchNorm layers stay frozen."""
    base = [l for l in model.layers if isinstance(l, keras.Model)]
    if not base:
        raise ValueError('No nested backbone found; is this a scratch model?')
    base = base[0]
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers[-n_layers:]:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    n = sum(1 for l in base.layers if l.trainable)
    tp = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f'unfroze {n} of {len(base.layers)} backbone layers | '
          f'{tp:,} trainable of {model.count_params():,} total')
    return model


# ---------------------------------------------------------------- train / eval
def train(model, train_ds, val_ds, epochs=20, lr=1e-3, tag='run',
          hierarchical=False, patience=6):
    os.makedirs(RESULTS, exist_ok=True)
    if hierarchical:
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss={'species': 'sparse_categorical_crossentropy',
                            'family': 'sparse_categorical_crossentropy'},
                      loss_weights={'species': 1.0, 'family': 0.3},
                      metrics={'species': 'accuracy', 'family': 'accuracy'})
        monitor, mode = 'val_species_accuracy', 'max'
    else:
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        monitor, mode = 'val_loss', 'min'
    cb = [keras.callbacks.ModelCheckpoint(f'{RESULTS}/{tag}.keras', monitor=monitor,
                                          mode=mode, save_best_only=True),
          keras.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                        patience=patience, restore_best_weights=True)]
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
    # Overwrite rather than append. Appending would silently concatenate two
    # unrelated runs if a cell is re-executed under the same tag.
    json.dump({k: list(v) for k, v in h.history.items()},
              open(f'{RESULTS}/{tag}_history.json', 'w'))
    return h


def _predict(model, ds, hierarchical):
    if hierarchical:
        y = np.concatenate([d['species'].numpy() for _, d in ds])
        yf = np.concatenate([d['family'].numpy() for _, d in ds])
        out = model.predict(ds, verbose=0)
        prob = out['species']
        return y, prob.argmax(1), prob.max(1), yf, out['family'].argmax(1)
    y = np.concatenate([yb.numpy() for _, yb in ds])
    prob = model.predict(ds, verbose=0)
    return y, prob.argmax(1), prob.max(1), None, None


def evaluate(model, ds, tag='run', hierarchical=False, split='test'):
    """split is recorded in the filename so validation and test results never mix."""
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, accuracy_score)
    names = class_names()
    y, p, conf, yf, pf = _predict(model, ds, hierarchical)
    rep = classification_report(y, p, target_names=names, output_dict=True,
                                zero_division=0)
    res = {'tag': tag, 'split': split,
           'accuracy': accuracy_score(y, p),
           'macro_f1': f1_score(y, p, average='macro', zero_division=0),
           'per_class': {n: rep[n] for n in names},
           'confusion': confusion_matrix(y, p).tolist(),
           'y_true': [int(v) for v in y], 'y_pred': [int(v) for v in p],
           # confidence lets the error gallery rank by how wrong the model was,
           # rather than showing whichever errors happen to come first
           'confidence': [round(float(v), 5) for v in conf]}
    if hierarchical:
        fmap, fams = family_map()
        res['fam_true'] = [int(v) for v in yf]
        res['fam_pred'] = [int(v) for v in pf]
        res['family_accuracy'] = float(accuracy_score(yf, pf))
        res['head_consistency'] = float(
            np.mean([fmap[int(s)] == int(f) for s, f in zip(p, pf)]))
    suffix = '' if split == 'test' else f'_{split}'
    json.dump(res, open(f'{RESULTS}/{tag}{suffix}_eval.json', 'w'), indent=1)
    msg = f'{tag} [{split}]: acc {res["accuracy"]:.4f} | macro F1 {res["macro_f1"]:.4f}'
    if hierarchical:
        msg += (f' | family acc {res["family_accuracy"]:.4f}'
                f' | head consistency {res["head_consistency"]:.4f}')
    print(msg)
    return res


def evaluate_saved(tags, ds, hierarchical=None, split='test'):
    """Reload checkpoints and score them. Keeps test evaluation in one place,
    after every configuration is locked."""
    out = []
    for t in tags:
        h = hierarchical if hierarchical is not None else t.startswith('C_')
        m = keras.models.load_model(f'{RESULTS}/{t}.keras')
        out.append(evaluate(m, ds, tag=t, hierarchical=h, split=split))
        del m
        keras.backend.clear_session()
    return out


# ---------------------------------------------------------------- analysis
def plot_history(histories, labels, title='', key='accuracy'):
    import matplotlib.pyplot as plt
    cols = plt.cm.viridis(np.linspace(0, .85, len(histories)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for h, lab, c in zip(histories, labels, cols):
        d = h.history if hasattr(h, 'history') else h
        acc = key if key in d else 'species_accuracy'
        e = np.arange(1, len(d['loss']) + 1)
        ax[0].plot(e, d['loss'], c=c, ls='--', lw=1)
        ax[0].plot(e, d['val_loss'], c=c, label=lab)
        ax[1].plot(e, d[acc], c=c, ls='--', lw=1)
        ax[1].plot(e, d['val_' + acc], c=c, label=lab)
    ax[0].set(xlabel='epoch', ylabel='loss', title='Loss (dashed = train)')
    ax[1].set(xlabel='epoch', ylabel='accuracy', title='Accuracy (dashed = train)')
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.show()


def summarize_runs(folder=RESULTS, split='test'):
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(f'{folder}/*_eval.json')):
        r = json.load(open(f))
        if r.get('split', 'test') != split:
            continue
        f1s = {n: v['f1-score'] for n, v in r['per_class'].items()}
        low = [v for n, v in f1s.items() if n in LOW_DATA]
        high = [v for n, v in f1s.items() if n not in LOW_DATA]
        row = {'run': r['tag'],
               'accuracy': round(r['accuracy'], 4),
               'macro_f1': round(r['macro_f1'], 4),
               'macro_f1_low': round(float(np.mean(low)), 4),
               'macro_f1_full': round(float(np.mean(high)), 4),
               'worst_class': min(f1s, key=f1s.get)}
        if 'head_consistency' in r:
            row['head_consistency'] = round(r['head_consistency'], 4)
        rows.append(row)
    if not rows:
        print(f'No {split} *_eval.json in {folder}/')
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('macro_f1', ascending=False)


def low_data_macro_f1(eval_json):
    r = json.load(open(eval_json))
    low = [v['f1-score'] for n, v in r['per_class'].items() if n in LOW_DATA]
    high = [v['f1-score'] for n, v in r['per_class'].items() if n not in LOW_DATA]
    out = {'tag': r['tag'], 'macro_f1_all16': round(r['macro_f1'], 4),
           'macro_f1_low_data': round(float(np.mean(low)), 4),
           'macro_f1_full_data': round(float(np.mean(high)), 4)}
    print(f"{out['tag']}: all16 {out['macro_f1_all16']:.4f} | "
          f"low-data {out['macro_f1_low_data']:.4f} | "
          f"full-data {out['macro_f1_full_data']:.4f}")
    return out


def paired_test(eval_a, eval_b, classes=None, verbose=True):
    """Exact McNemar on paired correctness. Tests ACCURACY, not macro F1.
    Use paired_bootstrap_f1 for the macro F1 difference."""
    from scipy.stats import binomtest
    ra, rb = json.load(open(eval_a)), json.load(open(eval_b))
    ya, yb = np.array(ra['y_true']), np.array(rb['y_true'])
    if not np.array_equal(ya, yb):
        raise ValueError('Runs scored on different items; not comparable.')
    pa, pb = np.array(ra['y_pred']), np.array(rb['y_pred'])
    mask = np.ones(len(ya), bool)
    if classes:
        names = class_names()
        keep = {names.index(c) for c in classes if c in names}
        mask = np.array([v in keep for v in ya])
    ca, cb = (pa == ya)[mask], (pb == yb)[mask]
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    n = int(mask.sum())
    pval = binomtest(c, b + c, 0.5).pvalue if b + c else 1.0
    if verbose:
        print(f'{ra["tag"]} vs {rb["tag"]}  (n={n}'
              f'{", low-data classes only" if classes else ""})')
        print(f'  accuracy {ca.mean():.4f} -> {cb.mean():.4f} '
              f'({(cb.mean()-ca.mean())*100:+.2f} pts)')
        print(f'  discordant: {b} only-A, {c} only-B '
              f'(discordance rate {(b+c)/n:.3f})')
        print(f'  exact McNemar p = {pval:.4f}')
    return {'model_a': ra['tag'], 'model_b': rb['tag'], 'n': n,
            'acc_a': float(ca.mean()), 'acc_b': float(cb.mean()),
            'discordant': b + c, 'only_a': b, 'only_b': c, 'p': float(pval)}


def paired_bootstrap_f1(eval_a, eval_b, classes=None, n_boot=2000, seed=SEED,
                        stratified=True):
    """Bootstrap CI for the macro F1 difference, resampling items in pairs.

    McNemar tests accuracy; our headline metric is macro F1. The same item
    indices are drawn for both models, which preserves the pairing.

    stratified=True resamples 60 observations within each species rather than
    freely across all 960. That matches the design, which fixes 60 per class, and
    matches macro F1, which weights every species equally."""
    from sklearn.metrics import f1_score
    ra, rb = json.load(open(eval_a)), json.load(open(eval_b))
    y = np.array(ra['y_true'])
    if not np.array_equal(y, np.array(rb['y_true'])):
        raise ValueError('Runs scored on different items.')
    pa, pb = np.array(ra['y_pred']), np.array(rb['y_pred'])
    if classes:
        names = class_names()
        keep = {names.index(c) for c in classes if c in names}
        m = np.array([v in keep for v in y])
        y, pa, pb = y[m], pa[m], pb[m]
    lab = sorted(set(y.tolist()))
    f = lambda t, q: f1_score(t, q, average='macro', labels=lab, zero_division=0)
    obs = f(y, pb) - f(y, pa)
    rng = np.random.default_rng(seed)
    groups = [np.where(y == c)[0] for c in lab] if stratified else None
    d = np.empty(n_boot)
    for i in range(n_boot):
        if stratified:
            k = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        else:
            k = rng.integers(0, len(y), len(y))
        d[i] = f(y[k], pb[k]) - f(y[k], pa[k])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'{ra["tag"]} -> {rb["tag"]}'
          f'{"  (low-data classes)" if classes else ""}'
          f'{"  [stratified by species]" if stratified else ""}')
    print(f'  delta macro F1 = {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]'
          f'{"  excludes zero" if lo * hi > 0 else "  includes zero"}')
    return {'delta': float(obs), 'lo': float(lo), 'hi': float(hi),
            'excludes_zero': bool(lo * hi > 0)}


def holm(results, alpha=0.05):
    """Holm-Bonferroni across a family of comparisons from paired_test."""
    import pandas as pd
    r = sorted(results, key=lambda d: d['p'])
    m, out, stopped = len(r), [], False
    for i, d in enumerate(r):
        thr = alpha / (m - i)
        rej = (d['p'] <= thr) and not stopped
        if not rej:
            stopped = True
        out.append({'comparison': f"{d['model_a']} vs {d['model_b']}",
                    'p': round(d['p'], 4), 'holm_threshold': round(thr, 4),
                    'significant': bool(rej)})
    return pd.DataFrame(out)


def family_confusion(eval_json, use_family_head=False):
    """Collapse species confusion to families.

    use_family_head=True uses the model's own family prediction (hierarchical
    runs only) instead of inferring family from the predicted species."""
    r = json.load(open(eval_json))
    fmap, fams = family_map()
    yt = np.array(r['y_true'])
    if use_family_head:
        if 'fam_pred' not in r:
            raise ValueError('No family head recorded for this run.')
        ft, fp = np.array(r['fam_true']), np.array(r['fam_pred'])
    else:
        ft = np.array([fmap[int(v)] for v in yt])
        fp = np.array([fmap[int(v)] for v in r['y_pred']])
    F = np.zeros((len(fams), len(fams)), int)
    for a, b in zip(ft, fp):
        F[a, b] += 1
    cm = np.array(r['confusion'])
    correct, total = int(np.trace(cm)), int(cm.sum())
    same = int(np.trace(F))
    src = 'family head' if use_family_head else 'inferred from species'
    print(f'{r["tag"]}  ({src})')
    print(f'  species correct     : {correct}/{total} = {correct/total:.3f}')
    print(f'  family correct      : {same}/{total} = {same/total:.3f}')
    print(f'  within-family errors: {same-correct} '
          f'({(same-correct)/max(total-correct,1):.1%} of all errors)')
    print(f'  cross-family errors : {total-same}')
    return F, fams


def error_taxonomy(eval_json):
    """Split errors three ways: same genus, same family but different genus, and
    different family. Congeners are the hardest case and the one a family-level
    hierarchy cannot help with, since they sit under the same family node."""
    r = json.load(open(eval_json))
    y, p = np.array(r['y_true']), np.array(r['y_pred'])
    fmap, _ = family_map()
    gmap = genus_map()
    wrong = np.where(y != p)[0]
    same_gen = [i for i in wrong if gmap[y[i]] == gmap[p[i]]]
    same_fam = [i for i in wrong
                if gmap[y[i]] != gmap[p[i]] and fmap[y[i]] == fmap[p[i]]]
    cross = [i for i in wrong if fmap[y[i]] != fmap[p[i]]]
    n = len(y)
    print(f'{r["tag"]}: {len(wrong)} errors out of {n} ({len(wrong)/n:.1%})')
    for lab, idx in [('same genus', same_gen),
                     ('same family, different genus', same_fam),
                     ('different family', cross)]:
        pct = len(idx) / max(len(wrong), 1)
        print(f'  {lab:30s} {len(idx):4d}  ({pct:.1%} of errors)')
    return {'same_genus': same_gen, 'same_family': same_fam, 'cross_family': cross}


def test_paths():
    """File paths for the test split, in the exact order the pipeline yields them.

    _ds preserves splits.csv row order and never shuffles validation or test, so
    this is a direct read. Callers should still assert alignment against y_true."""
    rows = [r for r in csv.DictReader(open(SPLITS)) if r['split'] == 'test']
    names = class_names()
    return ([r['file_path'] for r in rows],
            [names.index(r['scientific_name']) for r in rows])


def seed_spread(tags, folder=RESULTS):
    v = [(t, json.load(open(f'{folder}/{t}_eval.json'))) for t in tags]
    acc = np.array([r['accuracy'] for _, r in v])
    f1 = np.array([r['macro_f1'] for _, r in v])
    for t, r in v:
        print(f'  {t:32s} acc {r["accuracy"]:.4f}  macro F1 {r["macro_f1"]:.4f}')
    print(f'  accuracy  mean {acc.mean():.4f}  sd {acc.std(ddof=1):.4f}  '
          f'range {acc.max()-acc.min():.4f}')
    print(f'  macro F1  mean {f1.mean():.4f}  sd {f1.std(ddof=1):.4f}  '
          f'range {f1.max()-f1.min():.4f}')
    return {'acc_mean': float(acc.mean()), 'acc_sd': float(acc.std(ddof=1)),
            'f1_mean': float(f1.mean()), 'f1_sd': float(f1.std(ddof=1))}


def detectable_effect(n_per_class=60, baseline=0.70):
    se = np.sqrt(baseline * (1 - baseline) / n_per_class)
    print(f'n={n_per_class} per class, baseline {baseline:.0%}')
    print(f'  95% CI on a single class in isolation: +/- {1.96*se*100:.1f} points')
    print(f'  approximate unpaired 95% noise scale on a difference: '
          f'~{1.96*np.sqrt(2)*se*100:.1f} points')
    print('  (this is a noise scale, not a minimum detectable effect at a stated')
    print('   power; an 80% power calculation would use a larger multiplier)')
    print('  Paired tests on the SAME images are more sensitive than this, but by')
    print('  how much depends on the observed discordance rate, which cannot be')
    print('  known in advance. paired_test() reports it beside the p-value.')
