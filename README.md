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
         img
         its accuracy to test dataset is 91.02%
* #### train_model2.py: 
         deep learning model for Grad-CAM, Saliency Map 
         img
         its accuracy to test dataset is 91.34%
* #### train.py: 
         you can train your model for training dataset by this code   
         
* #### heatmap: 
         show heatmap for model1 and 2   
![heatmap](https://user-images.githubusercontent.com/30902020/89966513-4b96e800-dc8a-11ea-8c52-e12b8d87c6df.png)
<img src="/path/heatmap.png" width="450px" height="300px" title="px(픽셀) 크기 설정" alt="Heatmap"></img><br/>

