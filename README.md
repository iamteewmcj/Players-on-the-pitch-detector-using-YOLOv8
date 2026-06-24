# Football Player Detection & Team Clustering

A computer vision pipeline for football (soccer) broadcast footage that detects players, goalkeepers, referees, and the ball, then assigns players to their two teams using visual embeddings — all built on top of a custom-trained YOLOv8 detector.

## Overview

The project takes broadcast match frames and produces annotated output with:

- **Object detection** — players, goalkeepers, referees, and the ball, via a YOLOv8 model fine-tuned on football data.
- **Broadcast-style visualization** — each person marked with an ellipse at their feet (the look used in pro analytics overlays) rather than a plain box.
- **Team classification** — players split into their two teams without any hard-coded colors, using deep image embeddings instead of raw jersey color.

The pipeline is implemented from scratch in a single Colab notebook, using a pre-trained detector as the starting point and adding each capability as its own stage.

## Pipeline

1. **Player detection (YOLOv8, fine-tuned).** Starting from COCO-pretrained YOLOv8, the model is fine-tuned on the Roboflow `football-players-detection` dataset (372 images: train/val/test split) so it natively recognizes the four football classes — `ball`, `goalkeeper`, `player`, `referee` — instead of the generic COCO `person`/`sports ball`. Fine-tuning specifically solves the problem where a generic detector labels the referee and goalkeeper as ordinary players.

2. **Foot-point ellipse rendering.** Detections are drawn as open-arc ellipses at each player's ground-contact point (bottom-center of the box), color-coded by role, with the ball drawn as a separate marker.

3. **Team clustering (SigLIP → UMAP → K-Means).** Rather than averaging jersey color (which fails on low-contrast and green-on-grass kits), each player crop is passed through the **SigLIP** vision foundation model to produce a rich appearance embedding. **UMAP** reduces the embeddings to a low-dimensional space, and **K-Means (k=2)** splits the players into two teams. Clustering is run **per frame**, so each match finds its own two-team boundary; goalkeepers and referees are excluded from the clustering and shown as their own classes.

## Results

The fine-tuned detector was evaluated per-class on the held-out validation set:

| Class      | mAP@50 | mAP@50-95 | Recall |
|------------|--------|-----------|--------|
| player     | 0.97   | 0.64      | 0.96   |
| referee    | 0.83   | 0.46      | 0.76   |
| goalkeeper | 0.79   | 0.45      | 0.72   |
| ball       | 0.17   | 0.06      | 0.09   |

**Players, goalkeepers, and referees are detected reliably.** Separating the referee and goalkeeper into their own classes — the original failure case for a generic detector — works well after fine-tuning.

**Ball detection is the clear bottleneck.** At 0.09 recall the ball is missed in most frames. This is driven by its small size, motion blur, and a severe class imbalance (≈45 ball instances vs ≈970 player instances in validation). This is a known hard problem in football CV and is the natural target for future work.

**Team clustering** separates teams cleanly when the two kits are visually distinct (e.g. yellow vs red/white). It is strongest with the embedding-based approach and per-frame clustering; it degrades on low-contrast pairings (white vs dark) and on kits close to the pitch color (green on grass).

## Key findings

A few engineering takeaways surfaced during development:

- **Embeddings beat color for team assignment.** A baseline using grass-masked median jersey color collapsed on white-vs-dark and green kits. SigLIP embeddings, which capture full appearance rather than a single average color, cracked most of those cases.
- **Cluster per match, not globally.** Fitting one K-Means across frames from multiple matches assumes two universal teams, which is wrong. Running K-Means independently per frame fixed the frames that previously collapsed into a single cluster.
- **Fine-tuning fixes role confusion.** Generic detectors see only `person`; the fine-tuned model distinguishes player, goalkeeper, and referee, removing the need for downstream color-based filtering of officials.

## Tech stack

- **YOLOv8** (Ultralytics) — object detection, fine-tuned on football data
- **SigLIP** (Hugging Face Transformers) — player-crop image embeddings
- **UMAP** — dimensionality reduction
- **K-Means** (scikit-learn) — team clustering
- **OpenCV** — image processing and visualization
- Trained on Google Colab (Tesla T4 GPU)

## Dataset

[football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) from Roboflow Universe (CC BY 4.0), originally derived from the DFL — Bundesliga Data Shootout dataset.

## Future work

- Improve ball detection (higher inference resolution, oversampling ball instances, dedicated small-object handling).
- Player tracking across video frames (ByteTrack/DeepSORT) for persistent IDs, enabling speed, distance, and possession metrics.
- 2D pitch projection ("radar view") via pitch-keypoint detection and homography.
- Quantitative evaluation of team-clustering accuracy against hand-labeled teams.

## Acknowledgements

Inspired by Roboflow's open-source [sports](https://github.com/roboflow/sports) project. Detector built with Ultralytics YOLOv8; team clustering uses Google's SigLIP.
