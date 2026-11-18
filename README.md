# 👁️ PPA-ViT-Edge: An Edge-Deployable Vision Transformer for Peripapillary Atrophy Detection in Glaucoma Screening

**Official Open-Source Release for the Paper:**
> **"PPA-ViT-Edge: An Edge-Deployable Vision Transformer for Peripapillary Atrophy Detection in Glaucoma Screening"**
>
> *Juan Herón Rodríguez-Reséndiz & Jorge Francisco Martínez-Carballido (2027)*
---

This repository provides the curated cross-validation splits, fold reconstruction tools, and standalone edge inference code for **"PPA-ViT-Edge"**. Our framework combines anatomical peripapillary ring isolation (via Optic Disc core subtraction and Topological Smart Exclusion) with a customized **384x384** Vision Transformer (`M5`) featuring Learnable Attention Pooling, achieving **84.68% accuracy** and **~785 ms end-to-end latency** on a standalone **Raspberry Pi 5**.

## 📖 Citation

If you use this repository, the curated/processed images, or the exported ONNX models in your research, please cite our manuscript:

```bibtex
@article{Rodriguez2027PPAViTEdge,
  title={PPA-ViT-Edge: An Edge-Deployable Vision Transformer for Peripapillary Atrophy Detection in Glaucoma Screening},
  author={Rodríguez-Reséndiz, Juan Herón and Martínez-Carballido, Jorge Francisco},
  journal={Submission Pending},
  year={2027}
}
```

## 📁 Repository Structure & Release Assets

### In this Repository (Code & Splits)
* `lista_folds_dataset.csv`: Master cross-validation table detailing the exact `Train` and `Test` assignments across `Fold_1`, `Fold_2`, and `Fold_3` for all 1,807 curated images (1,026 `Non-PPA` and 781 `PPA`).
* `build_folds.py`: Automated utility script that reads `lista_folds_dataset.csv` and organizes the extracted dataset images into the 3-fold directory structure (`Fold_1`, `Fold_2`, and `Fold_3`).

### In [GitHub Releases](../../releases) (Dataset & Trained Weights)
Due to file size, the preprocessed image cohort and quantized model weights are hosted in the **[Releases](../../releases)** section:
1. **`Curated_Preprocessed_Datset.zip` (~454 MB)**: All 1,807 curated and preprocessed peripapillary ring images ready for training and evaluation.
2. **Quantized INT8 ONNX Weights (~85.24 MB each)**:
   * `PPA_ViT_Edge_M5_Fold1_int8.onnx`
   * `PPA_ViT_Edge_M5_Fold2_int8.onnx`
   * `PPA_ViT_Edge_M5_Fold3_int8.onnx`

---

## 🖼️ Overview of the Preprocessed Dataset

All images provided in `Curated_Preprocessed_Datset.zip` have already been processed through our two-phase anatomical pipeline:
1. **Phase 1 (Optic Disc Core Subtraction & Ring Isolation):** Suppresses the bright optic disc core (delineated via a domain-adapted SAM 2.1 Hiera-Tiny model) and isolates the surrounding peripapillary zone using proportional elliptical morphological operators. *(Note: The SAM 2.1 OD segmentation weights and preprocessing scripts are available upon reasonable request to the corresponding author).*
2. **Phase 2 (Topological Flood-Fill Smart Exclusion):** Applies $K=6$ red-channel intensity clustering with a radial flood-fill topological guard (`IsClosedRing`) to remove residual neuroretinal rim glare and camera artifacts without rupturing the annular ring continuity.

---

## 🛠️ Reconstructing the 3 Cross-Validation Folds

To recreate the exact `Fold_1`, `Fold_2`, and `Fold_3` directory tree used in the paper:

1. Download **`Curated_Preprocessed_Datset.zip`** from the [Releases](../../releases) page and extract it into the root of this repository.
2. Run the fold reconstruction script:
   ```bash
   python build_folds.py
3. The script will generate the `Dataset_3Folds/` directory containing `Fold_1`, `Fold_2`, and `Fold_3` with their respective `Train/` and `Test/` subdirectories (`PPA` and `Non-PPA`).

## 🍓 Edge Inference on Raspberry Pi 5 (ONNX Runtime)
The quantized PPA-ViT-Edge (M5) models run entirely on the ARM CPU of a Raspberry Pi 5 using onnxruntime without external accelerators.

1. Install Dependencies
   ```bash
   pip install onnxruntime opencv-python numpy

2. Standalone Inference Script (`raspi_inference.py`)
   Download any of the PPA_ViT_Edge_M5_FoldX_int8.onnx checkpoints and the preprocessed dataset from **[Releases](../../releases)**, and run the following script on any image from the extracted dataset:

```python
"""
PPA-ViT-Edge: Autonomous Peripapillary Atrophy Classification via ONNX Runtime
"""
import time
import cv2
import numpy as np
import onnxruntime as ort

# ===================== CONFIGURATION =====================
ONNX_MODEL_PATH = "PPA_ViT_Edge_M5_Fold1_int8.onnx"
IMAGE_PATH = "Dataset_3Folds/Fold_1/Test/PPA/1000_2_exclusion.png"  # Path to any preprocessed image
INPUT_SIZE = (384, 384)
CLASSES = ["Non-PPA", "PPA"]

# ImageNet Normalization Constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    
    # Convert BGR to RGB and resize to 384x384
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] and apply ImageNet standardization
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - MEAN) / STD
    
    # Transpose HWC -> NCHW (1, 3, 384, 384)
    tensor_nchw = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(tensor_nchw, axis=0)

def softmax(logits):
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / np.sum(exp_vals)

if __name__ == "__main__":
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Quad-core ARM Cortex-A76 on Raspberry Pi 5
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # End-to-End Latency Measurement (Preprocessing + Forward Pass)
    t0 = time.perf_counter()
    input_tensor = preprocess_image(IMAGE_PATH)
    t_infer_start = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    t_end = time.perf_counter()

    pure_latency_ms = (t_end - t_infer_start) * 1000
    e2e_latency_ms = (t_end - t0) * 1000

    logits = outputs[0][0]
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))

    print("=" * 50)
    print(f"Predicted Class    : {CLASSES[pred_idx]}")
    print(f"Confidence         : {probs[pred_idx] * 100:.2f}% (Non-PPA: {probs[0]*100:.1f}%, PPA: {probs[1]*100:.1f}%)")
    print(f"Pure Inference     : {pure_latency_ms:.2f} ms")
    print(f"End-to-End Latency : {e2e_latency_ms:.2f} ms")
    print("=" * 50)
```
## 📊 Dataset Attribution & References
The preprocessed images provided in `Curated_Preprocessed_Datset.zip` are derived from the publicly available **PPA data for glaucoma screening** dataset curated by the **King Abdullah International Medical Research Center (KAIMRC)** and hosted on the **Harvard Dataverse** under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

If you use the curated dataset splits or pretrained models in this repository, please cite the original dataset creators alongside our work:

Dataset Reference:

```bibtex
@dataset{PPAAlmazroa,
  author = {Almazroa, Ahmed},
  publisher = {Harvard Dataverse},
  title = {{PPA data for glaucoma screening}},
  year = {2022},
  version = {V1},
  doi = {10.7910/DVN/FITN2D},
}
```

Foundational Benchmark Publication:

```bibtex
@article{almansour2022peripapillary,
    doi = {10.1371/journal.pone.0275446},
    author = {Almansour, Abdullah AND Alawad, Mohammed AND Aljouie, Abdulrhman AND Almatar, Hessa AND Qureshi, Waseem AND Alabdulkader, Balsam AND Alkanhal, Norah AND Abdul, Wadood AND Almufarrej, Mansour AND Gangadharan, Shiji AND Aldebasi, Tariq AND Alsomaie, Barrak AND Almazroa, Ahmed},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Peripapillary atrophy classification using CNN deep learning for glaucoma screening},
    year = {2022},
    month = {10},
    volume = {17},
    url = {10.1371/journal.pone.0275446},
    pages = {1-25},
    number = {10}
}
```

## 📜 License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This project, including the fold reconstruction scripts, curated cross-validation splits (`lista_folds_dataset.csv`), and exported INT8 ONNX models (`PPA_ViT_Edge_M5_Fold1_int8.onnx`, `PPA_ViT_Edge_M5_Fold2_int8.onnx`, `PPA_ViT_Edge_M5_Fold3_int8.onnx`), is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

*(Note: The underlying retinal images in `Curated_Preprocessed_Datset.zip` are adapted from the PPA data for glaucoma screening dataset by Almazroa, Ahmed (2022) and remain governed by the original **CCO 1.0** license).*

**Under the following terms:**
- **Attribution:** You must give appropriate credit, provide a link to the license, and cite our paper alongside the original dataset creators (Almazroa, 2022; Almansour et al., 2022).
- **NonCommercial:** You may not use the material (code or trained ONNX models) for commercial purposes.
- **Derivatives Permitted:** You are free to remix, transform, adapt, and build upon the material for non-commercial research and academic purposes, provided you indicate if changes were made.
