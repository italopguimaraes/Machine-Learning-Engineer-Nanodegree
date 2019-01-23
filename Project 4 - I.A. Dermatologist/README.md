
[image1]: ./images/skin_disease_classes.png "Skin Disease Classes"
[image2]: ./images/cat_1.jpeg "Category 1 Rankings"
[image3]: ./images/cat_2.jpeg "Category 2 Rankings"
[image4]: ./images/cat_3.png "Category 3 Rankings"
[image5]: ./images/sample_ROC_curve.png "Sample ROC curve"
[image6]: ./images/sample_confusion_matrix.png "Sample confusion matrix"

# Dermatologist AI

## Introduction

In this project, a Convolutional Neural Network (CNN) was developed capable of visually diagnosing [melanoma](http://www.skincancer.org/skin-cancer-information/melanoma), the deadliest form of skin cancer. In particular, the algorithm is able to distinguish this malignant tumor and two types of benign skin lesions ([nevi](http://missinglink.ucsf.edu/lm/dermatologyglossary/nevus.html) and [seborrheic keratoses](https://www.aad.org/public/diseases/bumps-and-growths/seborrheic-keratoses)).

The data and objective are pulled from the [2017 ISIC Challenge on Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a).  As part of the challenge, participants were tasked to design an algorithm to diagnose skin lesion images as one of three different skin diseases (melanoma, nevus, or seborrheic keratosis).  In this project, you will create a model to generate your own predictions.

![Skin Disease Classes][image1]

## Software dependencies

Make sure the `sklearn`, `keras`, `opencv-python`, `numpy`, `matplotlib`, `pandas`, `tensorflow` and `jupyter notebook` are installed:

`conda install sklearn keras opencv-python numpy matplotlib pandas tensorflow jupyter notebook`

For more information see: 

[requirements](requirements.txt)

## Getting Started

1. Clone the [repository](https://github.com/Italo-Pereira-Guimaraes/Machine-Learning-Engineer-Nanodegree) and enter the `Project 4 - I.A. Dermatologist/` directory.   
```text
git clone https://github.com/Italo-Pereira-Guimaraes/Machine-Learning-Engineer-Nanodegree
cd Project 4 - I.A. Dermatologist
```
2. Download and unzip the data.
```text
mkdir train; mkdir valid; mkdir test
```
3. Download and unzip the [DataSet 1](https://drive.google.com/file/d/1BXf3O-C1ge6q33SEQNsFCzEDSSwSs5_p/view?usp=sharing) (5.10 GB).

4. Download and unzip the [DataSet 2](https://drive.google.com/file/d/1hr0mWJV-h4z56hbEOKNgsUWGPSoddXLU/view?usp=sharing) (6.48 GB).

5. Download and unzip the [Bottleneck Features](https://drive.google.com/file/d/1AbnTSVX9BLW6BisEKGFlc6nPwGGTdkQX/view?usp=sharing) (872 MB).

6. Download and unzip the [Saved Models](https://drive.google.com/file/d/1HRbB2UG_PO5hgorIpL8X2Wo4OEdiK1xO/view?usp=sharing) (230 MB).

7. Place the previously unpacked folders in the `Project 4 - I.A. Dermatologist / ` directory.

8. Run the `IA_dermatologist.ipynb` notebook and follow all steps.

## Create a Model

Use the training and validation data to train a model that can distinguish between the three different image classes.  (_After training, you will use the test images to gauge the performance of your model._)

If you would like to read more about some of the algorithms that were successful in this competition, please read [this article](https://arxiv.org/pdf/1710.05006.pdf) that discusses some of the best approaches.  A few of the corresponding research papers appear below.
- Matsunaga K, Hamada A, Minagawa A, Koga H. [“Image Classification of Melanoma, Nevus and Seborrheic Keratosis by Deep Neural Network Ensemble”](https://arxiv.org/ftp/arxiv/papers/1703/1703.03108.pdf). International Skin Imaging Collaboration (ISIC) 2017 Challenge at the International Symposium on Biomedical Imaging (ISBI). 
- Daz IG. [“Incorporating the Knowledge of Dermatologists to Convolutional Neural Networks for the Diagnosis of Skin Lesions”](https://arxiv.org/pdf/1703.01976.pdf). International Skin Imaging Collaboration (ISIC) 2017 Challenge at the International Symposium on Biomedical Imaging (ISBI). ([**github**](https://github.com/igondia/matconvnet-dermoscopy))
- Menegola A, Tavares J, Fornaciali M, Li LT, Avila S, Valle E. [“RECOD Titans at ISIC Challenge 2017”](https://arxiv.org/abs/1703.04819). International Skin Imaging Collaboration (ISIC)  2017 Challenge at the International Symposium on Biomedical Imaging (ISBI). ([**github**](https://github.com/learningtitans/isbi2017-part3))

While the original challenge provided additional data (such as the gender and age of the patients), we only provide the image data to you.  If you would like to download this additional patient data, you may do so at the competition [website](https://challenge.kitware.com/#phase/5840f53ccad3a51cc66c8dab).

All three of the above teams increased the number of images in the training set with additional data sources.  If you'd like to expand your training set, you are encouraged to begin with the [ISIC Archive](https://isic-archive.com/#images).

## Evaluation

Inspired by the ISIC challenge, your algorithm will be ranked according to three separate categories.

#### Category 1: ROC AUC for Melanoma Classification

In the first category, we will gauge the ability of your CNN to distinguish between malignant melanoma and the benign skin lesions (nevus, seborrheic keratosis) by calculating the area under the receiver operating characteristic curve ([ROC AUC](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)) corresponding to this binary classification task.

If you are unfamiliar with ROC (Receiver Operating Characteristic) curves and would like to learn more, you can check out the documentation in [scikit-learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py) or read [this Wikipedia article](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

The top scores (from the ISIC competition) in this category can be found in the image below.

![Category 1 Rankings][image2]

#### Category 2: ROC AUC for Melanocytic Classification

All of the skin lesions that we will examine are caused by abnormal growth of either [melanocytes](https://en.wikipedia.org/wiki/Melanocyte) or [keratinocytes](https://en.wikipedia.org/wiki/Keratinocyte), which are two different types of epidermal skin cells.  Melanomas and nevi are derived from melanocytes, whereas seborrheic keratoses are derived from keratinocytes. 

In the second category, we will test the ability of your CNN to distinuish between melanocytic and keratinocytic skin lesions by calculating the area under the receiver operating characteristic curve ([ROC AUC](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)) corresponding to this binary classification task.

The top scores in this category (from the ISIC competition) can be found in the image below.

![Category 2 Rankings][image3]

#### Category 3: Mean ROC AUC

In the third category, we will take the average of the ROC AUC values from the first two categories.

The top scores in this category (from the ISIC competition) can be found in the image below.

![Category 3 Rankings][image4]

## Getting your Results

Once you have trained your model, create a CSV file to store your test predictions.  Your file should have exactly 600 rows, each corresponding to a different test image, **plus** a header row.  You can find an example submission file (`sample_submission.csv`) in the repository.

Your file should have exactly 3 columns:
- `Id` - the file names of the test images (in the **same** order as the sample submission file)
- `task_1` - the model's predicted probability that the image (at the path in `Id`) depicts melanoma
- `task_2` - the model's predicted probability that the image (at the path in `Id`) depicts seborrheic keratosis

Once the CSV file is obtained, you will use the `get_results.py` file to score your submission.  To set up the environment to run this file, you need to create (and activate) an environment with Python 3.5 and a few pip-installable packages:
```text
conda create --name derm-ai python=3.5
source activate derm-ai
pip install -r requirements.txt
```

Once you have set up the environment, run the following command to see how the sample submission performed:
```text
python get_results.py sample_predictions.csv
```

Check the terminal output for the scores obtained in the three categories:
```text
Category 1 Score: 0.526
Category 2 Score: 0.606
Category 3 Score: 0.566
```

The corresponding **ROC curves** appear in a pop-up window, along with the **confusion matrix** corresponding to melanoma classification.  

![Sample ROC curve][image5]
![Sample confusion matrix][image6]

The code for generating the confusion matrix assumes that the threshold for classifying melanoma is set to 0.5.  To change this threshold, you need only supply an additional command-line argument when calling the `get_results.py` file.  For instance, to set the threshold at 0.4, you need only run:
```text
python get_results.py sample_predictions.csv 0.4
```

To test **your own** submission, change the code to instead include the path to **your** CSV file.
