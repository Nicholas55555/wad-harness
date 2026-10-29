# Pre-registration: O3 version B, a learned camera hazard estimator

Written 2026-10-08, before the training subset was downloaded or any model was trained.

## Data (never DAWN, never BDD100K val)

- BDD100K **train** split, **daytime** images only (DAWN is daytime; night is a separate axis).
- Five classes from the weather tag: clear; cloudy = overcast + partly cloudy; rain = rainy;
  snow = snowy; fog = foggy. Undefined weather is excluded.
- Up to 1,500 images per class, drawn at random with seed 0 from the daytime pool of that class;
  every daytime foggy image if fewer exist. 10% of the subset (stratified, seed 0) is held out
  for model selection; nothing else is used for selection.

## Model and training

- torchvision ResNet-18 with ImageNet weights, final layer replaced by a 5-way linear head.
- Input: the full frame resized to 224×400 (BDD100K's 16:9), ImageNet normalisation.
  Augmentation: horizontal flip and random-resized-crop (scale 0.6–1.0, aspect kept). No colour
  jitter (it would erase the contrast and saturation cues the task is about).
- AdamW, lr 1e-4, weight decay 1e-4, batch 32, label smoothing 0.1, class-balanced sampling,
  8 epochs, fp32, seed 0. The checkpoint with the best held-out balanced accuracy is kept.

## Hazard scalar and bins

- h_B(image) = p(rain) + p(snow) + p(fog), the predicted probability of adverse weather.
- Predicted bins: quartiles of h_B cut on each calibration half (as version A), test images
  assigned by the same cut points. Also reported: the argmax class against the true weather.
- h_B is a function of the image only (no detector output); the classifier never sees a DAWN
  or BDD100K-val image during training or selection.

## Evaluation (same procedure as phase 6a)

- DAWN with the run 1 detector (YOLOv8s @640, class loss, all labels) and with the run 2 primary
  (YOLOv8x @1280, group loss, clean labels): per-predicted-bin CRC at α ∈ {0.10, 0.20}, held-out
  miss fraction and FPPI per predicted bin and per true weather bin, FPPI gap to oracle bins,
  agreement with weather labels (contingency, adjusted Rand index), compared with version A.
- Success criteria (proposal slide 7): predicted-bin thresholds meet α in every weather bin
  where the global threshold fails, and cost at most 5 FPPI more than oracle bins.
- BDD100K val daytime: accuracy of the argmax class against the weather tag (sanity check).
