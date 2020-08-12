# git_pneumonia
 pneumonia classification

setting
-------
list
1.conda install -y opencv
2.conda install -y cmake
3.conda install -y -c conda-forge dlib
4.conda install -y pillow
5.conda install -y keras==2.1.6
6.conda install -y matplotlib
7.conda install -y tensorflow-gpu==1.14.0

Chest Xray Dataset link
-----------------------
<https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>
<https://radiopaedia.org/cases/childhood-pneumonia-1?lang=us>


code usage
----------
>train_model.py: train model for CAM

>train_model2.py: train model for Grad-CAM, Saliency Map

>train.py: training code

>heatmap: show heatmap for model1 and 2
