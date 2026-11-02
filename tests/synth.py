"""Synthetic segmentation cases with known ground truth and varying difficulty."""
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

SPACING_XYZ = (0.8, 0.8, 2.5)


def _ellipsoid(shape, center, radii):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (((zz - center[0]) / radii[0]) ** 2 + ((yy - center[1]) / radii[1]) ** 2
            + ((xx - center[2]) / radii[2]) ** 2) <= 1


def make_case(rng, shape=(16, 40, 40), difficulty=0.5, n_classes=3):
    """Label: 1 = organ, 2 = lesion inside the organ. Logits get worse with difficulty."""
    c = np.array(shape) / 2 + rng.normal(0, 1.5, 3)
    organ = _ellipsoid(shape, c, (4, 10, 7))
    lesion = _ellipsoid(shape, c + rng.normal(0, 1.0, 3), (2, 3, 3)) & organ
    label = np.zeros(shape, np.uint8)
    label[organ] = 1
    label[lesion] = 2
    # the model's belief: displaced and noisy for difficult cases
    shift = rng.normal(0, 2.5 * difficulty, 3)
    belief = ndimage.shift(label.astype(float), shift, order=0)
    logits = np.zeros((n_classes,) + shape, np.float32)
    for k in range(n_classes):
        logits[k] = 4.0 * (belief == k)
    noise = ndimage.gaussian_filter(rng.normal(0, 1, (n_classes,) + shape), sigma=(0, 1.5, 1.5, 1.5))
    logits += noise.astype(np.float32) * (6.0 * difficulty + 0.5)
    e = np.exp(logits - logits.max(0, keepdims=True))
    probs = e / e.sum(0, keepdims=True)
    return label, probs.astype(np.float32), logits


def write_image(arr, path, spacing=SPACING_XYZ):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, path, True)


def make_dataset(root, n=40, seed=0, layout="nnunet_v2", shape=(16, 40, 40)):
    """Writes probs/ and labels/ folders. Returns {case: difficulty}."""
    rng = np.random.default_rng(seed)
    pdir, ldir = os.path.join(root, "probs"), os.path.join(root, "labels")
    os.makedirs(pdir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    diffs = {}
    for i in range(n):
        cid = f"case_{i:03d}"
        d = float(rng.uniform(0.05, 1.0))
        label, probs, logits = make_case(rng, shape, d)
        diffs[cid] = d
        write_image(label, os.path.join(ldir, cid + ".nii.gz"))
        if layout == "nnunet_v2":
            np.savez_compressed(os.path.join(pdir, cid + ".npz"), probabilities=probs.astype(np.float16))
            write_image(probs.argmax(0).astype(np.uint8), os.path.join(pdir, cid + ".nii.gz"))
        elif layout == "logits":
            np.savez_compressed(os.path.join(pdir, cid + ".npz"), logits=logits)
        elif layout == "xyz":
            np.save(os.path.join(pdir, cid + ".npy"), np.transpose(probs, (0, 3, 2, 1)))
        elif layout == "single_channel":
            np.save(os.path.join(pdir, cid + ".npy"), probs[1] + probs[2])
        else:
            raise ValueError(layout)
    return diffs
