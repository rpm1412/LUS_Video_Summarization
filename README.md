# LUS_Video_Summarization
An implementation of the summarization algorithm in the paper titled - Unsupervised multi-latent space reinforcement learning framework for video summarization in ultrasound imaging. The link to the paper can be found [here.](https://arxiv.org/abs/2109.01309)

A video demonstration of the web-application deployment of the proposed video summarization model for Tele-medicine applications can be found at the website of the [Center for Computational Imaging](http://www.pulseecho.in/alus/video-summarization/) at IIT Palakkad. [Video here.](https://youtu.be/Th-XGQWRvpo)

The python script `vid_SAMGRAH_app.py` provides the webapp (GUI) summarization of the ultrasound videos along with machine classification scores and overlayed lung segmentations.
The python script `base_inference_code.py` provides the same summarization in a non gui manner. A link to the codeocean reproducible capsule is provided for running the code. [codeocean capsule here.](https://codeocean.com/capsule/8503804/tree/v1)

The `/data` folder contains 4 lung ultrasound videos that is used to demonstrate the summarization. The summarized videos can be found in `/summaryData` folder.

To run the webapp, please open the following folders in the working directory and enter: 
`streamlit run vid_SAMGRAH_app.py`

The folder format is as follows:
```
current directory              : main directory    
    |-> data                   : folder containing all original/raw lung ultrasound videos.    
    |-> summaryData            : folder for storing summarized videos that are generated.
    |-> modelWeights           : folder containing model weights.
        |-> decoder            : sub-folder in modelWeights for trained LSTM weights.
        |-> preTrainedEncoders : sub-folder in modelWeights for preTrained encoders.
    |-> encFeatsH5             : folder for storing generated h5 features from encoders (optional).
    |-> vid_SAMGRAH_app.py     : Webapp LUS video summarization
    |-> base_inference_code.py : LUS video summarization (Non-GUI)
```
A high level outline of the proposed system methodology is given in the figure below. Please refer the article for complete details.
![Outline of the Proposed System](https://raw.githubusercontent.com/rpm1412/LUS_Video_Summarization/main/fig/Overall_Framework.png)

#### ACADEMIC REFERENCES TO BE CITED:

1. Roshan P Mathews, Mahesh Raveendranatha Panicker, Abhilash R Hareendranathan, Yale Tung Chen, Jacob L Jaremko, Brian Buchanan, Kiran Vishnu Narayan, Kesavadas C, Greeta Mathews, “Unsupervised multi-latent space reinforcement learning framework for video summarization in ultrasound imaging”, accepted in IEEE Journal of Biomedical and Health Informatics. https://doi.org/10.1109/JBHI.2022.3208779

2. Roshan P Mathews, Mahesh Raveendranatha Panicker and Abhilash R Hareendranathan, “vid-SAMGRAH: A PyTorch framework for multi-latent space reinforcement learning driven video summarization in ultrasound imaging, in Elsevier Software Impacts, 100185 (2021). https://doi.org/10.1016/j.simpa.2021.100185 

3. Roshan P Mathews, Mahesh Raveendranatha Panicker, Abhilash R Hareendranathan, Yale Tung Chen, Jacob L Jaremko, Brian Buchanan, Kiran Vishnu Narayan, Kesavadas C, Greeta Mathews, “RL based Unsupervised Video Summarization for US Imaging”, in Proc. of MICCAI 3rd International Workshop of Advances in Simplifying Medical UltraSound (ASMUS). Sep. 2022. https://doi.org/10.1007/978-3-031-16902-1_3


