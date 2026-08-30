"""
fishlib_v2.py -- solo extension of fishlib.py (see README.md for attribution).

Adds two things the original course project's report explicitly left open:

1. Geo/temporal metadata fusion (RQ4). `_geo_temporal_encode()` turns each
   row's latitude/longitude/observed_on into a small fixed-width feature
   vector -- sinusoidal encoding for longitude and day-of-year (both wrap:
   -180 and +180 longitude are adjacent, Dec 31 and Jan 1 are adjacent, so a
   raw scalar would put them maximally far apart), linear normalization for
   latitude (which doesn't wrap). Missing values (citizen-science metadata
   is never 100% complete) get an explicit missing-indicator dimension
   rather than a NaN or a silently wrong imputed value.

2. A revised, symmetric augmentation policy (RQ5). The original's targeted
   augmentation applied extra contrast/brightness to only the 8 artificially
   thinned species, which the original report diagnosed as having shifted
   those classes' training distribution away from the untouched val/test
   distribution -- explaining the ~13pt F1 drop it caused. realistic_augment()
   tests that diagnosis directly by applying augmentation symmetrically
   across every class (no targeting), using transforms with a physical
   reason to appear in underwater citizen-science photos -- color-cast
   jitter, mild blur, occlusion -- rather than generic contrast/brightness.

Both additions are designed to slot into fishlib.py's existing conventions
(SEED, IMG_SIZE, BATCH, AUTOTUNE, BACKBONES, _trunk) rather than duplicate
them.
"""
import collections
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import fishlib

__version__ = '1.0'

# lat_norm, lon_sin, lon_cos, doy_sin, doy_cos, geo_missing, date_missing
META_DIM = 7


# ---------------------------------------------------------------- metadata

def _day_of_year(date_str):
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return dt.timetuple().tm_yday


def _geo_temporal_encode(rows):
    """rows: list of dicts with (possibly empty-string) 'latitude',
    'longitude', 'observed_on' keys, e.g. from csv.DictReader over splits.csv.

    Returns a float32 array of shape (len(rows), META_DIM). Always finite:
    missing or unparseable geo/date fields are zeroed and flagged via the
    missing-indicator dimensions rather than left as NaN, so the model can
    learn to rely on the image branch alone for those rows instead of being
    fed garbage."""
    n = len(rows)
    feats = np.zeros((n, META_DIM), dtype='float32')
    for i, r in enumerate(rows):
        lat_s = (r.get('latitude') or '').strip()
        lon_s = (r.get('longitude') or '').strip()
        date_s = (r.get('observed_on') or '').strip()

        lat = lon = None
        if lat_s and lon_s:
            try:
                lat, lon = float(lat_s), float(lon_s)
            except ValueError:
                lat = lon = None
        if lat is not None:
            feats[i, 0] = lat / 90.0
            feats[i, 1] = np.sin(2 * np.pi * lon / 360.0)
            feats[i, 2] = np.cos(2 * np.pi * lon / 360.0)
            feats[i, 5] = 0.0
        else:
            feats[i, 5] = 1.0

        doy = None
        if date_s:
            try:
                doy = _day_of_year(date_s)
            except ValueError:
                doy = None
        if doy is not None:
            feats[i, 3] = np.sin(2 * np.pi * doy / 365.0)
            feats[i, 4] = np.cos(2 * np.pi * doy / 365.0)
            feats[i, 6] = 0.0
        else:
            feats[i, 6] = 1.0

    return feats


# ---------------------------------------------------------------- augmentation

class ColorCastJitter(layers.Layer):
    """Per-image RGB channel shift simulating underwater blue/green color
    cast, which varies by water depth, turbidity, and time of day -- a real
    source of appearance variation in citizen-science underwater photos that
    generic contrast/brightness jitter doesn't specifically target."""

    def __init__(self, strength=20.0, **kw):
        super().__init__(**kw)
        self.strength = strength

    def call(self, x, training=None):
        if not training:
            return x
        b = tf.shape(x)[0]
        shift = tf.random.uniform([b, 1, 1, 3], -self.strength, self.strength)
        return tf.clip_by_value(x + shift, 0.0, 255.0)

    def get_config(self):
        c = super().get_config()
        c['strength'] = self.strength
        return c


class MildBlur(layers.Layer):
    """Applies a small fixed Gaussian-like blur to a random subset of images
    each batch, simulating the focus/motion softness common in handheld
    underwater photography -- unlike flip/rotation/zoom, which assume a
    sharp source image."""

    def __init__(self, prob=0.3, **kw):
        super().__init__(**kw)
        self.prob = prob
        k = tf.constant([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
        k = k / tf.reduce_sum(k)
        self._kernel = tf.reshape(k, [3, 3, 1, 1])

    def call(self, x, training=None):
        if not training:
            return x
        b = tf.shape(x)[0]
        mask = tf.random.uniform([b, 1, 1, 1]) < self.prob
        chans = tf.split(x, 3, axis=-1)
        blurred = tf.concat(
            [tf.nn.depthwise_conv2d(c, self._kernel, [1, 1, 1, 1], 'SAME')
             for c in chans], axis=-1)
        return tf.where(mask, blurred, x)

    def get_config(self):
        c = super().get_config()
        c['prob'] = self.prob
        return c


class RandomOcclusion(layers.Layer):
    """Zeroes a random rectangular patch per image, simulating partial
    occlusion by other fish, coral, or the frame edge -- common in
    unconstrained underwater photos, which is exactly the failure mode
    generic contrast/brightness jitter does nothing to prepare a model for."""

    def __init__(self, prob=0.3, max_frac=0.2, **kw):
        super().__init__(**kw)
        self.prob = prob
        self.max_frac = max_frac

    def call(self, x, training=None):
        if not training:
            return x
        shape = tf.shape(x)
        b, h, w = shape[0], shape[1], shape[2]
        h_f, w_f = tf.cast(h, tf.float32), tf.cast(w, tf.float32)

        apply = tf.random.uniform([b]) < self.prob
        ph = tf.cast(tf.random.uniform([b], 0.05, self.max_frac) * h_f, tf.int32)
        pw = tf.cast(tf.random.uniform([b], 0.05, self.max_frac) * w_f, tf.int32)
        y0 = tf.cast(tf.random.uniform([b]) * tf.cast(h - ph, tf.float32), tf.int32)
        x0 = tf.cast(tf.random.uniform([b]) * tf.cast(w - pw, tf.float32), tf.int32)

        rows = tf.range(h)[tf.newaxis, :, tf.newaxis, tf.newaxis]
        cols = tf.range(w)[tf.newaxis, tf.newaxis, :, tf.newaxis]
        y0b = y0[:, tf.newaxis, tf.newaxis, tf.newaxis]
        x0b = x0[:, tf.newaxis, tf.newaxis, tf.newaxis]
        phb = ph[:, tf.newaxis, tf.newaxis, tf.newaxis]
        pwb = pw[:, tf.newaxis, tf.newaxis, tf.newaxis]
        in_patch = ((rows >= y0b) & (rows < y0b + phb) &
                    (cols >= x0b) & (cols < x0b + pwb))
        apply_b = apply[:, tf.newaxis, tf.newaxis, tf.newaxis]
        keep = tf.cast(~(in_patch & apply_b), tf.float32)
        return x * keep

    def get_config(self):
        c = super().get_config()
        c['prob'] = self.prob
        c['max_frac'] = self.max_frac
        return c


def realistic_augment():
    """Symmetric augmentation: the same policy applied to every class alike,
    no LOW_DATA targeting. Includes fishlib's own base flip/rotation/zoom
    (default_augment(extra=False)) so this is a complete, standalone policy
    directly comparable to fishlib.default_augment(extra=True) (the
    original's targeted policy, reproduced for comparison in train_v2.ipynb)
    -- only the extra tier differs between the two."""
    return keras.Sequential([
        fishlib.default_augment(extra=False),
        ColorCastJitter(strength=20.0),
        MildBlur(prob=0.3),
        RandomOcclusion(prob=0.3, max_frac=0.2),
    ], name='realistic_augment')


# ---------------------------------------------------------------- data

def _ds_meta(rows, training, augment, branch):
    if branch not in ('image', 'metadata', 'both'):
        raise ValueError("branch must be 'image', 'metadata', or 'both'")

    names = fishlib.class_names()
    idx = {n: i for i, n in enumerate(names)}
    paths = [r['file_path'] for r in rows]
    ys = [idx[r['scientific_name']] for r in rows]
    meta = _geo_temporal_encode(rows)

    def load_img(p):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.image.resize(img, fishlib.IMG_SIZE)
        return tf.cast(img, tf.uint8)

    y_ds = tf.data.Dataset.from_tensor_slices(ys)

    if branch == 'image':
        path_ds = tf.data.Dataset.from_tensor_slices(paths)
        d = tf.data.Dataset.zip((path_ds, y_ds))
        d = d.map(lambda p, y: ({'image': load_img(p)}, y),
                  num_parallel_calls=fishlib.AUTOTUNE)
    elif branch == 'metadata':
        meta_ds = tf.data.Dataset.from_tensor_slices(meta)
        d = tf.data.Dataset.zip((meta_ds, y_ds))
        d = d.map(lambda m, y: ({'metadata': m}, y),
                  num_parallel_calls=fishlib.AUTOTUNE)
    else:  # both
        path_ds = tf.data.Dataset.from_tensor_slices(paths)
        meta_ds = tf.data.Dataset.from_tensor_slices(meta)
        d = tf.data.Dataset.zip((path_ds, meta_ds, y_ds))
        d = d.map(lambda p, m, y: ({'image': load_img(p), 'metadata': m}, y),
                  num_parallel_calls=fishlib.AUTOTUNE)

    d = d.cache()
    if training:
        d = d.shuffle(len(paths), seed=fishlib.SEED, reshuffle_each_iteration=True)
    d = d.batch(fishlib.BATCH)

    if branch in ('image', 'both'):
        def cast_image(xb, y):
            out = dict(xb)
            out['image'] = tf.cast(out['image'], tf.float32)
            return out, y
        d = d.map(cast_image, num_parallel_calls=fishlib.AUTOTUNE)

        if augment is not None:
            def apply_aug(xb, y):
                out = dict(xb)
                out['image'] = augment(out['image'], training=True)
                return out, y
            d = d.map(apply_aug, num_parallel_calls=fishlib.AUTOTUNE)

    return d.prefetch(fishlib.AUTOTUNE)


def load_splits_meta(condition='balanced', augment=None, branch='image'):
    """Like fishlib.load_splits, but yields (dict, label) batches keyed by
    branch ('image' -> {'image'}, 'metadata' -> {'metadata'}, 'both' ->
    both keys), with metadata drawn from _geo_temporal_encode(). Augmentation
    (if given) is applied to the image branch only, train split only."""
    rows = fishlib._rows(condition)
    g = collections.defaultdict(list)
    for r in rows:
        g[r['split']].append(r)
    return (_ds_meta(g['train'], True, augment, branch),
            _ds_meta(g['val'], False, None, branch),
            _ds_meta(g['test'], False, None, branch))


# ---------------------------------------------------------------- model

def build_fused_model(backbone='efficientnetb0', mode='frozen', n_classes=16,
                       branch='both', dense=256, dropout=0.4, seed=None):
    """Three architectures sharing everything (dense width, dropout,
    backbone/mode when the image branch is present) except which branch(es)
    feed the head -- so a difference in result is attributable to the input
    signal, not an incidental architecture change. branch='image' reduces to
    fishlib.build_model()'s own architecture; branch='metadata' is a small
    MLP on the 7-d encoding alone; branch='both' concatenates both embeddings
    before the shared dense head."""
    if branch not in ('image', 'metadata', 'both'):
        raise ValueError("branch must be 'image', 'metadata', or 'both'")
    seed = fishlib.SEED if seed is None else seed
    keras.utils.set_random_seed(seed)

    inputs = {}
    embeds = []

    if branch in ('image', 'both'):
        img_in = keras.Input(shape=(*fishlib.IMG_SIZE, 3), name='image')
        inputs['image'] = img_in
        embeds.append(fishlib._trunk(img_in, backbone, mode))

    if branch in ('metadata', 'both'):
        meta_in = keras.Input(shape=(META_DIM,), name='metadata')
        inputs['metadata'] = meta_in
        m = layers.Dense(32, activation='relu')(meta_in)
        m = layers.Dense(32, activation='relu')(m)
        embeds.append(m)

    x = embeds[0] if len(embeds) == 1 else layers.Concatenate()(embeds)
    x = layers.Dense(dense, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, out)
