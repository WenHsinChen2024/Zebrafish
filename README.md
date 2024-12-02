# Code and Datasets for Automated Zebrafish Spine Scoring System Based on Instance Segmentation

This repository contains the source code and datasets used in our research paper titled **"Automated Zebrafish Spine Scoring System Based on Instance Segmentation"**.  
The paper is currently under review for publication in *IEEE Access*. Once published, the DOI link will be provided here.

## Files

- `code/` - Contains the source code for data analysis and experiments.
- `datasets/` - Includes the datasets used in the research.
- `README.md` - This documentation file.

## How to Use

### 1.Clone the repository:
```bash
git clone https://github.com/WenHsinChen2024/Zebrafish.git
cd code/mmdetection
```
### 2.Install required dependencies:
```bash
pip install -r requirements.txt
```
### 3.Put the image to be recognized into the "sahi/fishimage" folder
### 4.Run the main script:
```bash
cd ..
cd sahi
python start.py
```
