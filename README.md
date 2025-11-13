# FDCFusion
Frequency-Domain CAFormer-CNN fusion network for infrared and visible image fusion

TNO datasets：https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029
MSRS datasets: https://github.com/Linfeng-Tang/MSRS?tab=readme-ov-file
M3FD datasets：https://github.com/JinyuanLiu-CV/TarDAL
RoadScene datasets:https://github.com/hanna-xu/RoadScene

Overview：
This project implements a frequency-domain caformer-cnn fusion network for infrared and visible image fusion.
It contains two main components:Fusion Model — extracts and fuses features from infrared and visible images;Illumination Classification Model — predicts illumination conditions to guide adaptive fusion.The project supports model training, testing, and quantitative evaluation on datasets such as MSRS, TNO, and RoadScene.

How to Run：
1️⃣ Train the Illumination Classification Model
python train_illum_cls.py
This trains the illumination perception network to classify lighting conditions (e.g., bright vs. dim scenes).
2️⃣ Train the Fusion Model
python train_fusion_model.py
This trains the image fusion model using infrared and visible image pairs.
You can configure training parameters (e.g., learning rate, epochs, data paths) inside the script.
3️⃣ Test the Fusion Model
python test_fusion_model.py
This runs inference on the test dataset and outputs the fused images.
4️⃣ Evaluate Fusion Performance
python evaluate.py
This script computes quantitative metrics such as SSIM, PSNR, VIF, and Entropy to assess fusion quality.
🧩 Dependencies：
Python ≥ 3.8
PyTorch ≥ 1.10
NumPy
OpenCV
tqdm
matplotlib
Install all dependencies via:
pip install -r requirements.txt
