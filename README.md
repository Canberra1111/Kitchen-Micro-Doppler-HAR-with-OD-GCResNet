# Kitchen-Micro-Doppler-HAR-with-OD-GCResNet

This repository includes our BumblebeeKitchen dataset and codes of the paper titled "Kitchen Micro-Doppler Human Activity Recognition using OD-GCResNet" submitted to IEEE ICC 2025 
(2025 IEEE International Conference on Communications).
![Overview](https://raw.githubusercontent.com/Canberra1111/Kitchen-Micro-Doppler-HAR-with-OD-GCResNet/main/Radar_System_Architecture.png)


## Project Overview

This project addresses the growing need for non-invasive monitoring solutions for elderly individuals, particularly focusing on kitchen activities, which pose unique safety challenges. We introduce **OD-GCResNet**, a novel deep learning model for kitchen activity recognition based on micro-Doppler signatures. 
The model combines advanced convolution techniques to accurately detect subtle micro-Doppler variations in real-world environments. We achieve a classification accuracy of **99.61%** on our dataset, outperforming baseline models. We also validate the model on the UORED-VAFCLS dataset and reached a classification accuracy of **99.29%**.
![Model Structure](https://raw.githubusercontent.com/Canberra1111/Kitchen-Micro-Doppler-HAR-with-OD-GCResNet/main/Model_Structure.png)





## Dataset

1. **Our BumblebeeKitchen Dataset**:
   - The dataset in this study is available in the `dataset.zip` file in this repository.
   
2. **Public Dataset**:
   - Access the public dataset(UORED-VAFCLS dataset) used for validation from the following link:
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
   git lfs clone https://github.com/Canberra1111/Kitchen-Micro-Doppler-HAR-with-OD-GC

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

version https://git-lfs.github.com/spec/v1
oid sha256:4eb13b6ff7ea44858c0e59172d9bdcbe33802ff6bfee1ac0ad3e268fe95a3d67
size 44

## Citations
If you used our work or found it helpful, please use the following citation:
   ```bash
   @inproceedings{pei-etal-2024-odgcresnet,
     title = "OD-GCResNet: A Deep Learning Model for Kitchen Activity Recognition Using Micro-Doppler Signatures",
     author = "Pei, Yinan and Hu, Shaohua and Cao, Junjia and Tan, Mingshu and Li, Zhuyi and Luo, Fei and Li, Anna",
     booktitle = "Proceedings of the 2025 IEEE International Conference on Communications (Under Review)",
     year = "2024",
     address = "Beijing, China",
     url = "https://github.com/Canberra1111/Kitchen-Micro-Doppler-HAR-with-OD-GCResNet",
     abstract = "As the global aging population grows, the need for non-invasive, reliable monitoring solutions for elderly individuals living alone becomes    urgent. Kitchen activities, a high-risk area in homes, pose unique safety challenges. However, existing human activity recognition methods still struggle    with accuracy, and few studies specifically address these challenges in kitchen environments. This paper introduces OD-GCResNet, a novel hybrid deep         learning model for kitchen activity recognition based on micro-Doppler signatures. The proposed model combines Omni-Dimensional Dynamic Convolution with     Recursive Gated Convolution to enhance global feature interactions and adaptive attention, allowing for accurate detection of subtle micro-Doppler           variations in complex, real-world environments. We validate OD-GCResNet on our collected kitchen activity dataset and achieve a classification accuracy 
   of 99.61%, outperforming baseline models. This work represents a significant step forward in non-contact, privacy-preserving safety monitoring solutions 
   for elderly care.",
   }



