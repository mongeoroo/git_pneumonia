# git_pneumonia
 pneumonia binary classification

setting
-------
list   
>conda install -y opencv   
>conda install -y cmake   
>conda install -y -c conda-forge dlib   
>conda install -y pillow   
>conda install -y keras==2.1.6   
>conda install -y matplotlib   
>conda install -y tensorflow-gpu==1.14.0   

Chest Xray Dataset link
-----------------------
><https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>   
><https://radiopaedia.org/cases/childhood-pneumonia-1?lang=us>


code usage
----------
* #### train_model.py:   
         deep learning model for CAM.   
         this model consist of DenseNetwork169 and SENet         
         its accuracy to test dataset is 91.02%
![model1](https://github.com/mongeoroo/git_pneumonia/blob/master/image/model1_architecture.png)
* #### train_model2.py: 
         deep learning model for Grad-CAM, Saliency Map 
         img
         its accuracy to test dataset is 91.34%
* #### train.py: 
         you can train your model for training dataset by this code   
         
* #### heatmap: 
         show heatmap for model1 and 2   
         the heatmap applied to "childhood-pneumonia-1.jpg" is like below.
         and you can apply your data to this model by changing the path which is in the main block.
![heatmap](https://github.com/mongeoroo/git_pneumonia/blob/master/image/heatmap.png)


