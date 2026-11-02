"""Reading probability maps and labels in many layouts, and writing output maps."""
import glob
import logging
import os
import pickle
import re

import numpy as np
import SimpleITK as sitk

log = logging.getLogger("conseg")

IMAGE_EXTS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ARRAY_EXTS = (".npz", ".npy")
PIXEL_UNIT_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------------- discovery
def split_ext(path):
    name = os.path.basename(path)
    low = name.lower()
    for e in IMAGE_EXTS + ARRAY_EXTS:
        if low.endswith(e):
            return name[: -len(e)], e
    return os.path.splitext(name)


def _expand(paths):
    files = []
    for p in paths:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        if not hits:
            raise FileNotFoundError(f"Nothing found at: {p}")
        for h in hits:
            if os.path.isdir(h):
                files += [os.path.join(h, f) for f in sorted(os.listdir(h))]
            elif os.path.isfile(h):
                files.append(h)
            else:
                raise FileNotFoundError(f"Nothing found at: {h}")
    return files


def index_probabilities(paths, suffix=None):
    """case_id -> probability file. Array files (.npz/.npy) are preferred over images,
    because nnU-Net prediction folders also contain the hard masks as .nii.gz."""
    files = _expand(paths)
    if suffix:
        cand = [f for f in files if f.lower().endswith(suffix.lower())]
    else:
        arrays = [f for f in files if split_ext(f)[1] in ARRAY_EXTS]
        cand = arrays if arrays else [f for f in files if split_ext(f)[1] in IMAGE_EXTS]
    out = {}
    for f in cand:
        cid, _ = split_ext(f)
        if cid in out:
            raise ValueError(f"Duplicate case '{cid}': {out[cid]} and {f}")
        out[cid] = f
    if not out:
        raise FileNotFoundError(f"No probability files found in {paths} (suffix={suffix})")
    return out


def index_images(paths, suffix=None, strip_channel=False):
    """case_id -> image/label file. strip_channel removes nnU-Net channel suffixes (_0000)."""
    out = {}
    for f in _expand(paths):
        cid, ext = split_ext(f)
        if suffix:
            if not f.lower().endswith(suffix.lower()):
                continue
        elif ext not in IMAGE_EXTS:
            continue
        if strip_channel:
            m = re.match(r"^(.*)_\d{4}$", cid)
            if m:
                cid = m.group(1)
        out.setdefault(cid, f)
    return out


def pair_cases(prob_index, label_index=None, required=True, what="cases"):
    if label_index is None:
        return {c: (p, None) for c, p in sorted(prob_index.items())}
    common = sorted(set(prob_index) & set(label_index))
    no_label = sorted(set(prob_index) - set(label_index))
    if no_label:
        log.warning("%d %s have probabilities but no label and are skipped (e.g. %s)",
                    len(no_label), what, ", ".join(no_label[:3]))
    if required and not common:
        raise ValueError(
            f"No {what} could be paired by case ID. Example probability IDs: {sorted(prob_index)[:3]}; "
            f"example label IDs: {sorted(label_index)[:3]}. File names (without extension) must match.")
    return {c: (prob_index[c], label_index[c]) for c in common}


# ----------------------------------------------------------------------------- reading
def load_prob_file(path, fmt="auto"):
    cid, ext = split_ext(path)
    meta = {"path": path, "format": None, "pkl": None, "image": None}
    if ext == ".npz":
        z = np.load(path)
        if "probabilities" in z.files:
            arr, meta["format"] = z["probabilities"], "nnunet_v2"
        elif "softmax" in z.files:
            arr, meta["format"] = z["softmax"], "nnunet_v1"
        else:
            key = z.files[0]
            arr, meta["format"] = z[key], "npz"
            meta["key"] = key
    elif ext == ".npy":
        arr, meta["format"] = np.load(path), "npy"
    else:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        if img.GetNumberOfComponentsPerPixel() > 1:
            arr = np.moveaxis(arr, -1, 0)
        meta["format"], meta["image"] = "image", img
    if fmt not in ("auto", None):
        meta["format"] = fmt
    pkl = os.path.join(os.path.dirname(path), cid + ".pkl")
    meta["pkl"] = pkl if os.path.isfile(pkl) else None
    return np.asarray(arr, dtype=np.float32), meta


def load_label(path, binarize_255=True):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if img.GetNumberOfComponentsPerPixel() > 1:     # RGB label image: use first channel
        arr = arr[..., 0]
    arr = arr.astype(np.int32)
    if binarize_255:
        u = np.unique(arr)
        if 255 in u and set(u.tolist()) <= {0, 255}:
            arr = (arr > 127).astype(np.int32)
    spacing = tuple(float(s) for s in reversed(img.GetSpacing()))   # array (z, y, x) order
    units = "px" if split_ext(path)[1] in PIXEL_UNIT_EXTS else "mm"
    return arr, img, spacing, units


_warned = set()


def _warn_once(key, msg, *args):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def _uncrop_nnunet_v1(arr, pkl_path, label_shape, has_ch):
    with open(pkl_path, "rb") as f:
        props = pickle.load(f)
    bbox = props.get("crop_bbox")
    orig = props.get("original_size_of_raw_data")
    if bbox is None or orig is None or tuple(int(s) for s in orig) != tuple(label_shape):
        raise ValueError(f"Cannot restore the nnU-Net v1 crop using {pkl_path}")
    full = np.zeros(((arr.shape[0],) if has_ch else ()) + tuple(label_shape), np.float32)
    if has_ch:
        full[0] = 1.0                                  # outside the crop = background
    sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
    full[(slice(None),) + sl if has_ch else sl] = arr
    _warn_once("v1crop", "nnU-Net v1 probabilities were cropped; restored to the full image using the .pkl crop box.")
    return full


def _drop_singletons(arr, d):
    """Remove size-1 axes after the first one until the array fits d spatial axes plus channels
    (nnU-Net v2 saves 2D natural-image probabilities as C x 1 x H x W)."""
    while arr.ndim > d + 1:
        ones = [ax for ax in range(1, arr.ndim) if arr.shape[ax] == 1]
        if not ones:
            break
        arr = np.squeeze(arr, ones[0])
    return arr


def standardize(arr, meta, label_shape=None, expect=None, array_order="auto"):
    """Return (array, has_channels, transposed) with channels first and spatial axes in label (zyx) order.
    label_shape: spatial shape of the reference mask, if available.
    expect: settings learned at calibration (spatial_ndim, has_channels, n_channels, transposed)."""
    if label_shape is not None:
        d = len(label_shape)
        arr = _drop_singletons(arr, d)
        if arr.ndim == d + 1:
            cand = [ax for ax in range(arr.ndim)
                    if sorted(np.delete(np.array(arr.shape), ax).tolist()) == sorted(label_shape)]
            ax = 0 if (0 in cand or not cand) else cand[-1]
            arr, has_ch = np.moveaxis(arr, ax, 0), True
        elif arr.ndim == d:
            has_ch = False
        else:
            raise ValueError(f"Probability array {arr.shape} does not fit a {d}-D label of shape {label_shape}")
    elif expect is not None:
        d = expect["spatial_ndim"]
        arr = _drop_singletons(arr, d)
        has_ch = arr.ndim == d + 1
        if has_ch and arr.shape[0] != expect["n_channels"]:
            axes = [ax for ax in range(arr.ndim) if arr.shape[ax] == expect["n_channels"]]
            arr = np.moveaxis(arr, axes[-1] if axes else 0, 0)
    else:
        raise ValueError("Need a label shape or calibration settings to interpret the probability array")

    spatial = tuple(arr.shape[1:] if has_ch else arr.shape)
    transposed = False

    def _transpose(a):
        k = len(spatial)
        perm = list(range(k))[::-1]
        return np.transpose(a, [0] + [p + 1 for p in perm]) if has_ch else np.transpose(a, perm)

    if label_shape is not None:
        if array_order == "xyz":
            arr, transposed = _transpose(arr), True
        elif spatial != tuple(label_shape):
            if array_order == "auto" and spatial[::-1] == tuple(label_shape):
                arr, transposed = _transpose(arr), True
                _warn_once("xyz", "Probability axes are in (x, y, z) order; transposed to match the labels.")
            elif meta.get("format") == "nnunet_v1" and meta.get("pkl"):
                arr = _uncrop_nnunet_v1(arr, meta["pkl"], label_shape, has_ch)
            else:
                raise ValueError(
                    f"{os.path.basename(meta['path'])}: probability shape {spatial} does not match label shape "
                    f"{tuple(label_shape)}. If the model reoriented the images (eg, nibabel RAS reorientation), "
                    f"export probabilities in the original image orientation.")
    elif expect is not None and expect.get("transposed"):
        arr, transposed = _transpose(arr), True
    return arr, has_ch, transposed


def to_probabilities(arr, has_ch, kind="auto"):
    """Convert to probabilities in [0, 1].
    Returns (array, kind after conversion: softmax|sigmoid, raw input kind)."""
    if kind == "auto":
        lo, hi = float(arr.min()), float(arr.max())
        multi = has_ch and arr.shape[0] > 1
        if lo >= -1e-4 and hi <= 1 + 1e-3:
            kind = "sigmoid"
            if multi:
                flat = arr.reshape(arr.shape[0], -1)
                idx = np.linspace(0, flat.shape[1] - 1, min(5000, flat.shape[1])).astype(int)
                if np.allclose(flat[:, idx].sum(0), 1.0, atol=0.05):
                    kind = "softmax"
        else:
            kind = "logits_softmax" if multi else "logits_sigmoid"
            _warn_once("logits", "Input values outside [0, 1]: treated as logits (%s).", kind)
    raw = kind
    if kind == "logits_softmax":
        a = arr - arr.max(0, keepdims=True)
        np.exp(a, out=a)
        arr, kind = a / a.sum(0, keepdims=True), "softmax"
    elif kind == "logits_sigmoid":
        arr, kind = 1.0 / (1.0 + np.exp(-arr)), "sigmoid"
    return np.clip(arr, 0.0, 1.0).astype(np.float32), kind, raw


# ----------------------------------------------------------------------------- geometry and writing
def geometry_reference(case_id, prob_meta, label_img=None, image_path=None):
    """An object describing output geometry: a SimpleITK image, a dict, or None."""
    if label_img is not None:
        return label_img
    if image_path:
        return sitk.ReadImage(image_path)
    folder = os.path.dirname(prob_meta["path"])
    for ext in (".nii.gz", ".nii", ".mha", ".nrrd"):
        sib = os.path.join(folder, case_id + ext)
        if os.path.isfile(sib) and sib != prob_meta["path"]:
            return sitk.ReadImage(sib)
    if prob_meta.get("image") is not None:
        im = prob_meta["image"]
        return {"spacing": im.GetSpacing(), "origin": im.GetOrigin(), "direction": im.GetDirection()}
    if prob_meta.get("pkl"):
        try:
            with open(prob_meta["pkl"], "rb") as f:
                props = pickle.load(f)
            st = props.get("sitk_stuff") or {}
            if st:
                return {"spacing": st["spacing"], "origin": st["origin"], "direction": st["direction"]}
            if "itk_spacing" in props:
                return {"spacing": props["itk_spacing"], "origin": props["itk_origin"],
                        "direction": props["itk_direction"]}
        except Exception as e:   # pragma: no cover
            log.debug("could not read %s: %s", prob_meta["pkl"], e)
    return None


def write_map(arr, ref, path_noext):
    """Write a map with the geometry of ref (NIfTI), or as .npy when no geometry is known."""
    if ref is None:
        np.save(path_noext + ".npy", arr)
        return path_noext + ".npy"
    img = sitk.GetImageFromArray(arr)
    if isinstance(ref, dict):
        img.SetSpacing(tuple(ref["spacing"]))
        img.SetOrigin(tuple(ref["origin"]))
        img.SetDirection(tuple(ref["direction"]))
    elif tuple(img.GetSize()) == tuple(ref.GetSize()):
        img.CopyInformation(ref)
    else:
        _warn_once("geom", "Output map size differs from the reference image; geometry not copied.")
    out = path_noext + ".nii.gz"
    sitk.WriteImage(img, out, True)
    return out
