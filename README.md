# git_pneumonia
##pneumonia binary classification

##setting
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
<img src="https://github.com/mongeoroo/git_pneumonia/blob/master/image/model1_architecture.png" width="600px" height="100px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img><br/>
* #### train_model2.py: 
         deep learning model for Grad-CAM, Saliency Map 
         its accuracy to test dataset is 91.34%
<img src="https://github.com/mongeoroo/git_pneumonia/blob/master/image/model2_architecture.png" width="730px" height="100px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img><br/>

* #### train.py: 
         you can train your model for training dataset by this code   
         
* #### heatmap: 
         show heatmap for model 1 and 2   
         the heatmap applied to "childhood-pneumonia-1.jpg" is like below.
         and you can apply your data to this model by changing the path which is in the main block.
![heatmap](https://github.com/mongeoroo/git_pneumonia/blob/master/image/heatmap.png)


