# Accurate and Efficient Human ActivityRecognition Through Semi-Supervised DeepLearning
## Introduction
 In recent years, deep learning has been widely used
for human activity recognition (HAR) based on wearable sensor
data due to its excellent performance. However, deep learning
approaches often need a large amount of labeled data to train the
model. The collection of huge amounts of labeled activity data is
usually labor-intensive and even cost-prohibitive, limiting the prac-
tical use of these methods. An effective way to address this issue is
to use semi-supervised learning methods. Yet, most existing semi-
supervised approaches fail to fully make use of unlabeled data to
improve deep learning models’ accuracy and generalizability. In
this work, we propose an innovative Adaptive Confidence Semi-
supervised HAR (ACS-HAR) method that can dynamically adjust
the selective confidence threshold based on the model’s learning
state, thereby making more efficient use of unlabeled data. Exper-
imental results on six public datasets demonstrate that our method outperforms traditional supervised methods by
up to 5.20% accuracy and state-of-the-art semi-supervised methods by up to 3.30% accuracy

## Installation   
To start up, you need to install some packages. Our implementation is based on PyTorch. We recommend using conda to create the environment and install dependencies and all the requirements are listed in `environment3090.yml`.
## Datasets
A total of six datasets are used, and this repository utilizes four datasets: UCI-HAR, WISDM, mHealth, and UCI-HAPT. For the PAMAP2 and USC-HAD datasets, you can download the raw datasets from the official websites, and then preprocess the datasets further using the scripts in the data_process folder within the data directory. The command is as follows:`python pamap_preprocess.py`
## Training
After the environment and the datasets are configured. You can run the code as follows:

```python
python ACS-HAR.py --c config/ACS-HAR/ACS_HAR_uci6_0.yaml
