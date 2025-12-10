## ___***[NeurIPS 2025]TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels***___

![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='https://arxiv.org/abs/2512.08358'><img src='https://img.shields.io/badge/arXiv-2512.08358-b31b1b.svg'></a> &nbsp;
 <a href='https://igl-hkust.github.io/TrackingWorld.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;

**Authors**:
[*Jiahao Lu*](https://github.com/jiah-cloud), [*Weitao Xiong*](https://openreview.net/profile?id=~Weitao_Xiong1),
[*Jiacheng Deng*](), [*Peng Li*](https://scholar.google.com/citations?user=8eTLCkwAAAAJ&hl=zh-CN), [*Tianyu Huang*](https://scholar.google.com/citations?view_op=list_works&hl=en&user=nhbSplwAAAAJ), [*Zhiyang Dou*](https://frank-zy-dou.github.io/), [*Cheng Lin*](https://clinplayer.github.io/), [*Sai-Kit Yeung*](https://saikit.org/index.html), [*Yuan Liu*](https://liuyuan-pal.github.io/) NeurIPS, 2025

---

**TrackingWorld** is a novel approach for **dense, world-centric 3D tracking** from **monocular videos**. Our method estimates accurate camera poses and disentangles 3D trajectories of both static and dynamic components — not limited to a single foreground object. It supports **dense tracking of nearly all pixels**, enabling robust 3D scene understanding from monocular inputs.

---

### 🖼️ Teaser

![Watch the teaser](assets/vis1_00.png)

---

## ⚙️ Setup and Installation

TrackingWorld relies on several **visual foundation model** repositories included as submodules for comprehensive preprocessing.

### 1\. Cloning the Repository

Use the `--recursive` flag to clone the main repository and all necessary submodules:

```bash
git clone --recursive https://github.com/IGL-HKUST/TrackingWorld.git
cd TrackingWorld
```

### 2\. Environment Setup

An installation script is provided and tested with **CUDA Toolkit 12.1** and **Python 3.10**.

```bash
conda create -n uni4d python=3.10
conda activate uni4d
bash scripts/install.sh
```

### 3\. Downloading Weights

Download the necessary model weights for the visual foundation models used in the pipeline:

```bash
bash scripts/download.sh
```

### 4\. OpenAI API Key (For Preprocessing)

Our initial preprocessing involves using **GPT** via the [OpenAI API](https://platform.openai.com/) (minimal credit usage expected). Please set your API key as an environment variable in a `.env` file:

```bash
echo "OPENAI_API_KEY=sk-your_api_key_here" > .env
```

Find your API key [here](https://platform.openai.com/api-keys).

---

That's a great structure for a GitHub README demonstration section\! It's clear, comprehensive, and logically separates the execution command from the resulting file organization.

Here is a slightly enhanced and polished version of the demonstration section, integrating the file paths into the directory structure for better visual clarity and ensuring all tags are fully described.

## 🚀 Demo

We've included the **`dog` sequence** from the DAVIS dataset as a demonstration. You can run the entire processing pipeline using the following convenience script:

```bash
bash scripts/demo.sh
```

---

### 📁 Output Structure

The demo generates a comprehensive set of intermediate and final results within the **`data/demo_data/`** directory. The files showcase the progression from foundational model outputs to the final 4D representation. You can also download a preprocessed version of the results [here](https://drive.google.com/file/d/133ZezKfsJJY8-gG4CoN9KEeHaDs0TDOH/view?usp=sharing).

```
data/demo_data/
└── dog/                               # 🐾 Demo Sequence Name (e.g., DAVIS 'dog')
    ├── color/                         # Original RGB Images
    │   └── 00000.jpg, ...             # Sequential RGB frames
    │
    ├── deva/                          # DEVA Model Outputs (Video Segmentation)
    │   └── pred.json, Annotations/, ...
    │
    ├── ram/                           # RAM Model Outputs (Image Tagging)
    │   └── tags.json                  # Contains RAM tags, GPT filtering results, and detected classes
    │
    ├── unidepth/                      # Depth Estimation Results
    │   ├── depth.npy                  # Raw depth maps
    │   └── intrinsics.npy             # Camera intrinsic parameters
    │
    ├── gsm2/                          # GSM2 Model Outputs (Instance/Semantic Segmentation)
    │   └── mask/, vis/, ...
    │
    ├── densetrack3d_efep/             # DenseTrack3D / CoTracker Outputs
    │   └── results.npz                # Dense tracklet data
    │
    └── uni4d/                         # Final Uni4D Reconstruction Outputs
        └── experiment_name/           # Experiment Name (e.g., base_delta_ds2)
            ├── fused_track_4d_full.npz    # 🔑 Fused 4D Representation (Main Output)
            └── training_info.log          # Training metadata
```

---

### ✨ Visualization

To visualize the dense 4D trajectories and the reconstructed scene, run the provided visualization script, pointing it to the main output file:

```python
python visualizer/vis_trackingworld.py --filepath data/demo_data/dog/uni4d/base_delta_ds2/fused_track_4d_full.npz
```

This visualization helps interpret the **world-centric motion** and **disentangled trajectories** generated by TrackingWorld.

![Watch the video](assets/TrackingWorld_video.gif)
## 📜 To-Do List

We plan to release more features and data soon.

* [X] Release demo code
* [ ] Provide evaluation benchmark and metrics

---

## 📝 Citation

If you find **TrackingWorld** useful for your research or applications, please consider citing our paper:

```bibtex
@inproceedings{
    lu2025trackingworld,
    title={TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels},
    author={Jiahao Lu and Weitao Xiong and Jiacheng Deng and Peng Li and Tianyu Huang and Zhiyang Dou and Cheng Lin and Sai-Kit Yeung and Yuan Liu},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={[https://openreview.net/forum?id=vDV912fa3t](https://openreview.net/forum?id=vDV912fa3t)}
}
```
## 🤝 Acknowledgements
Our codebase is based on [Uni4D](https://github.com/Davidyao99/uni4d). Our preprocessing relies on [DELTA](https://github.com/snap-research/DELTA_densetrack3d), [CotrackerV3](https://github.com/facebookresearch/co-tracker), [Unidepth](https://github.com/lpiccinelli-eth/UniDepth), [Tracking-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA), [Grounded-Sam-2](https://github.com/IDEA-Research/Grounded-SAM-2), and [Recognize-Anything](https://github.com/xinyu1205/recognize-anything). We thank the authors for their excellent work!