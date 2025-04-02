# CMSR
the code for cross-modal saliency for the immersive videos


**1. Download dataset**
***
The multi-modal saliency database  can be downloaded for xx.
The directory structure of the dataset is as follow.
<pre>
└── the multi-modal saliency dataset
    ├── saliency
    ├── fixation  
    ├── video_frames  
    ├── the original videos
</pre>
**2. The training and  testing code**

 
 **if you want to train the model, you need to download the dataset and specify the address of the traing set, then execute the following  instruction
```python
 python main_nowandb.py

** if you want to test the pretrained model, you just need to note the training section in the 'main_nowandb.py'.
The predict.py  will generate the saliency map and the evaluation_saliencymap.py will calculate the different evaluation metrics and save them as an excel file.
