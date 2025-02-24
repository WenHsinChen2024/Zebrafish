# **Code and Datasets for Automated Zebrafish Spine Scoring System Based on Instance Segmentation**
This repository contains the source code and datasets used in our research paper: **"[Automated Zebrafish Spine Scoring System Based on Instance Segmentation](https://ieeexplore.ieee.org/document/10849560)"**  
📖 Published in IEEE Access | DOI: [10.1109/ACCESS.2025.3532680](https://doi.org/10.1109/ACCESS.2025.3532680)  


## **Files**
```
├── code/              # Contains the source code for data analysis and experiments.
├── datasets/          # Includes the datasets used in the research.
├── README.md          # This documentation file.
```
## **Zebrafish Dataset**

The **Zebrafish Spine Dataset** is a high-quality dataset of zebrafish images used for instance segmentation and spine condition assessment. This dataset was used in our research paper.

### **Citation**  

If you use this dataset, please cite the following paper:

```
@article{10849560,
  author={Chen, Wen-Hsin and Kuo, Tien-Ying and Wei, Yu-Jen and Ho, Cheng-Jung and Lin, Ming-der and Chen, Huan and Lin, Wen-Ying},
  journal={IEEE Access}, 
  title={Automated Zebrafish Spine Scoring System Based on Instance Segmentation}, 
  year={2025},
  volume={13},
  number={},
  pages={18814-18826},
  keywords={Feature extraction;Accuracy;Micromechanical devices;Classification algorithms;Object recognition;Instance segmentation;Deep learning;Computer architecture;Proposals;Prediction algorithms;Deep learning;machine learning;object segmentation;image analysis},
  doi={10.1109/ACCESS.2025.3532680}}
```

---

### **License & Usage Policy**  
- This dataset is provided for **academic research purposes only**.  
- Users **must cite our paper** when using this dataset in publications.  
- If any of the images belong to you and you would like them removed, please contact us.

---

### **Dataset Overview**  

The **Zebrafish Dataset** is designed for medical imaging and computer vision research, particularly for **automated spine scoring** using deep learning models.  

#### **Dataset Structure**
The dataset is organized as follows:  

```
datasets/
├── Fish/
        ├── images/          # High-resolution zebrafish images
        ├── annotations/     # COCO format annotation files
├── Spine/
        ├── images/          # High-resolution zebrafish spine images
        ├── annotations/     # COCO format annotation files

```

#### **Data Splits**
There are 232 images for both the **Fish** category and the **Spine** category.
| Dataset Split  | Number of Images | Purpose |
|---------------|----------------|---------|
| Training Set  | 199            | Model training |
| Validation Set| 17            | Model tuning & evaluation |
| Test Set      | 16            | Final evaluation |

---

### **Data Access**
To download the dataset, clone this repository:  

```bash
git clone https://github.com/WenHsinChen2024/Zebrafish.git
```


---

## **Automated Zebrafish Spine Scoring System**  

This dataset is accompanied by a deep learning-based **automated spine scoring system**, built using **PyTorch** and **MMDetection**.  

### **Installation**  

**Using Conda (Recommended)**  

```bash
conda create --name zebrafish_env python=3.8 -y
conda activate zebrafish_env
```
**Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/)**

**Install Dependencies for MMDetection**  

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -r requirements.txt
```

**Install required Dependencies**
```bash
cd sahi
pip install -r requirements.txt
```

### **Run the Spine Scoring System**  

1. Place the zebrafish images that need to be identified in the `sahi/fishimage/` folder.  
2. Run the following command:  

```bash
cd code/sahi
python start.py
```

---

## **Contact**  
For any inquiries, please reach out to:  
📧 [t111318537@ntut.edu.tw]  

---
