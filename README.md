# garbage_classification
## Abstract
This project is to implement garbage classification by DeepLearning and the datasets are from HuaWei cloud and web. In this project you can classify a garbage only by its one picture or name. What's more, I use Inception-Resnet-V2 model to classify images and BiLSTM model plus Conv1D model to classify names in Keras. 
## How to get start
### Structure
#### by content
In this document you are able to classify a garbage by its name.   
1.datasets document contains garbage classification content datasets from web.  
2.contents_data_setting.py can deal with the datasets in ordedr feed them into network.  
3.model_training.py allows you to train the network in the datasets you have deal with.  
4.model_testing.py allows you to load and use a retrained network to classify a garbage by its name.  
5.settings.py contains some hyper-parameters used in model_training.py and so on.
#### by image
In this document you are able to classify a garbage by its one image.   
1.images_data_setting.py can deal with the datasets in ordedr feed them into network.  
2.model_training.py allows you to train the network in the datasets you have deal with.  
3.model_testing.py allows you to load and use a retrained network to classify a garbage by its one image.  
4.settings.py contains some hyper-parameters used in model_training.py and so on.
### Training by yourself
Just run model_training.py directly is ok and when the training process is over, the retrained-model will be saved automatically.
## PS
1.Before you run py, please make sure the paths in them is available for your computer.  
2.The images datasets should be downloaded by yourself in https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/garbage_classify/dataset/garbage_classify.zip
