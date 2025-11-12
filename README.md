# Kitchen-Micro-Doppler-HAR-with-OD-GCResNet

This repository includes our BumblebeeKitchen dataset and codes of the paper titled "A Deep Learning Model for Kitchen Activity Recognition using Micro-Doppler Signatures" **published by IEEE ICC 2025 
(2025 IEEE International Conference on Communications).**
![Overview](https://raw.githubusercontent.com/Canberra1111/Kitchen-Micro-Doppler-HAR-with-OD-GCResNet/main/Radar_System_Architecture.png)


## Project Overview

This project addresses the growing need for non-invasive monitoring solutions for elderly individuals, particularly focusing on kitchen activities, which pose unique safety challenges. We introduce **OD-GCResNet**, a novel deep learning model for kitchen activity recognition based on micro-Doppler signatures. 
The model combines advanced convolution techniques to accurately detect subtle micro-Doppler variations in real-world environments. We achieve a classification accuracy of **99.61%** on our dataset, outperforming baseline models. We also validate the model on the UORED-VAFCLS dataset and reach a classification accuracy of **99.29%**.
![Model Structure](https://raw.githubusercontent.com/Canberra1111/Kitchen-Micro-Doppler-HAR-with-OD-GCResNet/main/Model_Structure.png)





## Dataset

1. **Our BumblebeeKitchen Dataset**:
   - The dataset in this study is available in the `dataset.zip` file in this repository.
   
2. **Public Dataset**:
   - Access the public dataset (UORED-VAFCLS dataset) used for validation from the following link:
     - [Public Dataset on Mendeley](https://data.mendeley.com/datasets/y2px5tg92h/5)

## Dependencies

Compatible with PyTorch >= 2.0+CUDA 11.8 (lower CUDA versions may also work), and Python 3.x.
Dependencies can be installed using requirements.txt.


## Installation

1. Install Git LFS

   ```bash
   git lfs install

2. Clone the repository using Git LFS

   ```bash
   git lfs clone https://github.com/Canberra1111/Kitchen-Micro-Doppler-HAR-with-OD-GCResNet

3. Install the required dependencies

   ```bash
   pip install -r requirements.txt

## Usage

1. Main Experiment 

   ```bash
   python main.py

2. Model Validation on UORED-VAFCLS Dataset

   ```bash
   python validation.py

## Citations
If you used our work or found it helpful, please use the following citation:
   ```bash
   @inproceedings{pei-etal-2025-odgcresnet,
     title = "OD-GCResNet: A Deep Learning Model for Kitchen Activity Recognition Using Micro-Doppler Signatures",
     author = "Pei, Yinan and Hu, Shaohua and Cao, Junjia and Tan, Mingshu and Li, Zhuyi and Luo, Fei and Li, Anna",
     booktitle = "Proceedings of the 2025 IEEE International Conference on Communications",
     year = "2025",
     address = "Beijing, China",
     url = "https://github.com/Canberra1111/Kitchen-Micro-Doppler-HAR-with-OD-GCResNet",
   }
