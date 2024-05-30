# A set of simple studies and solutions

## Predict whether a mammogram mass is benign or malignant

* **Data**: The dataset can be found from University of Irvine: [Mammographic Mass](https://archive.ics.uci.edu/dataset/161/mammographic+mass). 
* **Goal**: Build a Multi-Layer Perceptron and train it to classify masses as benign or malignant based on its features.
* **Challenges**: The data needs to be cleaned; many rows contain missing data, and there may be erroneous data identifiable as outliers as well.
* **Approach**:

    * Review data quality, and missing data. Drop if not a lot of records are wrong
    * Transform the data to be usable by sklearn using numpy

See personal notebook in [mammogram_mass folder](https://github.com/jbcodeforce/ML-studies/tree/master/notebooks/mammogram_mass/personal-study.ipynb)

## Computer vision with PyTorch: classify sushi, pizza and steak

* **Data**: The food 101 dataset from [PyTorch vision](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* **Goal**: Develop a NN to classify images
* **Challenges**: The number of layers
* **Approach**: Develop a basic NN and then compare it with existing CNN

???- demo "Demonstration with pytorch scripts"
    * Create a virtual env and install requirements under the pytorch folder; `pip install -r requirements.txt`
    * Under computer vision, load the data sets locally: `python  prepare_image_dataset.py --classes sushi,steak,pizza`
    * Do a simple classification using a Tiny VGG: `python classify_food.py`
    * Use transfer learning (see [explanations here](../ml/deep-learning.md#transfer-learning)): `python transfer_learning.py`


## Other use case

* **Data**: 
* **Goal**: 
* **Challenges**: 
* **Approach**:
# A set of simple studies and solutions

## Predict whether a mammogram mass is benign or malignant

* **Data**: The dataset can be found from University of Irvine: [Mammographic Mass](https://archive.ics.uci.edu/dataset/161/mammographic+mass). 
* **Goal**: Build a Multi-Layer Perceptron and train it to classify masses as benign or malignant based on its features.
* **Challenges**: The data needs to be cleaned; many rows contain missing data, and there may be erroneous data identifiable as outliers as well.
* **Approach**:

    * Review data quality, and missing data. Drop if not a lot of records are wrong
    * Transform the data to be usable by sklearn using numpy

See personal notebook in [mammogram_mass folder](https://github.com/jbcodeforce/ML-studies/tree/master/notebooks/mammogram_mass/personal-study.ipynb)

## Computer vision with PyTorch: classify sushi, pizza and steak

* **Data**: The food 101 dataset from [PyTorch vision](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* **Goal**: Develop a NN to classify images
* **Challenges**: The number of layers
* **Approach**: Develop a basic NN and then compare it with existing CNN

???- demo "Demonstration with pytorch scripts"
    * Create a virtual env and install requirements under the pytorch folder; `pip install -r requirements.txt`
    * Under computer vision, load the data sets locally: `python  prepare_image_dataset.py --classes sushi,steak,pizza`
    * Do a simple classification using a Tiny VGG: `python classify_food.py`
    * Use transfer learning (see [explanations here](../ml/deep-learning.md#transfer-learning)): `python transfer_learning.py`


## Other use case

* **Data**: 
* **Goal**: 
* **Challenges**: 
* **Approach**: