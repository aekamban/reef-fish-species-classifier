"""
Sanity-checks fishlib_v2.py against synthetic data. This sandbox has no GPU
and no iNaturalist access, so this cannot validate real accuracy -- it only
proves the data pipeline, metadata encoding, augmentation, and the three
model variants (image/metadata/fused) build and run a forward+backward pass
without shape or wiring bugs, before real training happens in Colab.
"""
import csv
import os
import random
import shutil
import tempfile

import numpy as np
from PIL import Image

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import fishlib
import fishlib_v2 as v2

SPECIES = [
    ('Thalassoma pavo', 'Labridae', 'Thalassoma'),
    ('Thalassoma bifasciatum', 'Labridae', 'Thalassoma'),
    ('Abudefduf saxatilis', 'Pomacentridae', 'Abudefduf'),
    ('Chaetodon auriga', 'Chaetodontidae', 'Chaetodon'),
]


def make_synthetic_dataset(tmpdir, n_per_species=12):
    img_dir = os.path.join(tmpdir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    rng = random.Random(fishlib.SEED)
    rows = []
    for sci, fam, genus in SPECIES:
        for i in range(n_per_species):
            arr = (rng.random() * 255 * np.ones((32, 32, 3))).astype('uint8')
            path = os.path.join(img_dir, f'{sci.replace(" ", "_")}_{i}.jpg')
            Image.fromarray(arr).save(path)
            has_geo = rng.random() > 0.15
            has_date = rng.random() > 0.15
            rows.append({
                'scientific_name': sci, 'family': fam, 'genus': genus,
                'file_path': path,
                'latitude': str(rng.uniform(-30, 30)) if has_geo else '',
                'longitude': str(rng.uniform(-90, 90)) if has_geo else '',
                'observed_on': '2026-0%d-%02d' % (rng.randint(1, 9), rng.randint(1, 28)) if has_date else '',
                'split': 'train' if i < n_per_species - 4 else ('val' if i < n_per_species - 2 else 'test'),
                'train_low_data': '',
            })
    splits_path = os.path.join(tmpdir, 'splits.csv')
    with open(splits_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return splits_path


def main():
    tmpdir = tempfile.mkdtemp(prefix='fishlib_v2_test_')
    try:
        fishlib.SPLITS = make_synthetic_dataset(tmpdir)
        print('== class_names / family_map ==')
        print(fishlib.class_names())
        print(fishlib.family_map())

        print('\n== metadata encoding shape/values ==')
        rows = fishlib._rows('balanced')
        feats = v2._geo_temporal_encode(rows)
        assert feats.shape == (len(rows), v2.META_DIM), feats.shape
        assert np.isfinite(feats).all()
        print('feats shape', feats.shape, 'sample row', feats[0])

        for branch in ('image', 'metadata', 'both'):
            print(f'\n== branch={branch} ==')
            aug = v2.realistic_augment() if branch in ('image', 'both') else None
            train_ds, val_ds, test_ds = v2.load_splits_meta(
                'balanced', augment=aug, branch=branch)
            xb, yb = next(iter(train_ds))
            print('batch input keys', list(xb.keys()),
                  {k: v.shape for k, v in xb.items()}, 'y', yb.shape)

            model = v2.build_fused_model(
                backbone='mobilenetv2', mode='frozen', n_classes=4, branch=branch)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            h = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)
            loss = h.history['loss'][0]
            assert np.isfinite(loss), f'{branch}: non-finite loss'
            print(f'{branch}: 1-epoch train loss = {loss:.4f} (finite, no shape errors)')

        print('\nAll fishlib_v2 synthetic checks passed.')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
