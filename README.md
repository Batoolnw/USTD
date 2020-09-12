# USTD
The source code of the USTD (The Untold Sentiments at Tourists Destination) project.
NOTE: We are still extending the USTD  as much as possible
# The Untold Sentiments at Tourists Destination
USTD (The Untold Sentiments at Tourists Destination) .

## Abstract
NEOM will be recognized as a global industry leader for next-generation entertainment delivering a breadth of unique experiences that overwhelm the senses we plan to implement an intelligent system that will assist in determining the expression of NEOM visitors during their accommodation. Our system relies on the visitor’s picture taken by NEOM’s cameras, visitor device camera, visitor’s library, or visitor’s snaps shared in the SnapMap or any other social media platform to evaluate the satisfaction using the face expression analyze. The face expression analyze function along with the results will be available online to help NEOM’s administration, entrepreneurs, visitors to improve their services and/or open new services in places that have the highest happiness rate. As well as for the visitors by knowing where is the happiest places.  

## Dataset
[fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) is the dataset I chose, which is anounced in Kaggle competition in 2013.

## Environment
I provide my work environment for references.

### Hadware
CPU : i5-6500  
GPU : nvidia GTX 960 2G  
RAM : 8G  

### Software
OS  : Ubuntu 16.04  
Keras 1.2.0  
scikit-learn 0.18.1  
opencv 3.1.0  

## Installation
I strongly recommend you to use [`Anaconda`](https://www.continuum.io/downloads), which is a package manager and provides python virtual environment.  
After you install Anaconda, you can create a virtual environment with python 3.4.
```
conda create -n env-name python=3.4
```
you can also check if your env. has been created by,
```
conda info --envs
```
You should activate your virtual environment in different way corresponding to your operating system.
For example, In Ubuntu, you can activate your virtual environment by,
```
source activate env-name
```
And,
```
source deactivate
```
to exit the virtual environment.

The following instructions will lead you to install dependencies, and I suggest you to fllow the order.
#### Install scikit-learn
```
conda install scikit-learn
```
#### Install OpenCV
Note that the version `Anaconda` provided may not be the latest one.
```
conda install opencv
```
If you fail to install opencv due to python version conflicts, try this command instead,
```
conda install -c menpo opencv3=3.1.0
```
the version 3.1.0 can be replaced with the lateset one, but in this project, I use `opencv 3.1.0`.
#### Install Keras
Keras is a high-level wrapper of Theano and Tensorflow, it provides friendly APIs to manipulate several kinds of deep learning models.
```
pip install --upgrade keras
```
#### Install pandas and h5py
`pandas` can help you to preprocess data if you want train your own deep learning model.
```
conda install pandas
```
`h5py` is used to save weights of pre-trained models.
```
conda install h5py
```
#### Configuration
Before executing this project, you should make `Keras` use `Theano` backend by modifying configuration file in
```
~/.keras/keras.json
```
If it doesn't exist, you can create a new one, and then change the content to

if you use kears 1 :

```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```
if you use kears 2 :

```
{
    "image_data_format": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

## Usage
### Simple facial expression detection
After installing dependencies, you can move to `webcam` directory and simply type this command,
```
python webcam_detection.py
```
and the system will start detecting user's emotions and print results to the console.  
### Affecting computing system
If you want to combine facail expression detection and speech recognition to generate a completed sentence with appropriate emoticons,
you should install an additional dependency.
```
pip install SpeechRecognition
```
After installing the above library, you can type this to lauch the detector.
```
python gen_sentence_with_emoticons.py
```
Launch the system and input "y" to start the detection, then you can speek something with facial expression to try to acquire a sentence with emoticons.

