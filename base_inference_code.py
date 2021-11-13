#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 00:00:00 2021

@author: Roshan P Mathews & Mahesh R Panicker

This code is the implementation of the summarization algorithm in the paper 
titled - Unsupervised multi-latent space reinforcement learning framework for video summarization in ultrasound imaging 
that takes lung US vidoes and output summarized videos with machine classification score and lung segmentations.  

The folder format to run the app is as follows.

current directory : main directory
        
-> data : folder for all raw lung ultrasound videos.
        
-> summaryData : folder for storing summarized videos are generated.
        
-> modelWeights : folder for all model weights.
            
--> decoder : sub-folder in modelWeights for trained LSTM weights.
            
--> preTrainedEncoders : sub-folder in modelWeights for preTrained encoders.
        
-> encFeatsH5 : folder for storing generated h5 features from encoders.
"""

#%% Importing Librarires

import os
import numpy as np
import h5py
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import argparse
import torch.nn.functional as F

torch.manual_seed(1)

#%% Encoders

# Classifier Network
class LUSNet(nn.Module):
    def __init__(self):
        super(LUSNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #64,128,128
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2), #64,64,64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #128,64,64
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2), #128,32,32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), #256,32,32
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2), #256,16,16

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), #512,16,16
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2), #512,8,8

            nn.AdaptiveAvgPool2d(output_size=1), #globalAvgPooling
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 1)
            )
          
    def forward(self, x):
        feats = self.features(x)
        clsf = self.classifier(feats)
        return clsf, feats

# AutoEncoder Network
class LUSAENet(nn.Module):
    def __init__(self,latentDim = 512):
        super(LUSAENet, self).__init__()
        
        self.latentDim = latentDim
       
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2), #16,32,32
            nn.SELU(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2), #32,15,15
            nn.SELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2), #64,7,7
            nn.SELU()
        )
        self.flatten = nn.Flatten()

        self.code = nn.Linear(7*7*64, self.latentDim)

        self.toDecoder = nn.Linear(self.latentDim, 7*7*64)
        # reshaping done in forward
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2), #32,15,15
            nn.SELU(),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2), #16,32,32
            nn.SELU(),

            nn.ConvTranspose2d(16, 1, kernel_size=8, stride=4, padding=2), #1,128,128
            nn.ReLU()
        )

    def forward(self, x):
      x_single = torch.unsqueeze(x[:,0,:,:],1)

      feats = self.encoder(x_single)
      featsFlatten = self.flatten(feats)
      
      codeSequence = self.code(featsFlatten)

      codeToDecoder = self.toDecoder(codeSequence)
      batchSizeUsed = codeToDecoder.size(0)
      codeReshapeToDecoder = codeToDecoder.view((batchSizeUsed,64,7,7))
      
      predictedImg = self.decoder(codeReshapeToDecoder)
      reconImg = torch.cat((predictedImg,predictedImg,predictedImg), 1)
      
      return reconImg, codeSequence

#%% Loading Encoders

def loadModelFile(filepath):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath,  map_location=torch.device(device))
    
    print(checkpoint['model'])
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()

    return model

#%% Image Augmentations

def testImgAugUNET(imgFrame):
    
    TEST_TFMS = albu.Compose([
        albu.Resize(height=128, width=128),
        albu.Normalize(),
        ToTensorV2()
    ])

    img = imgFrame.squeeze().numpy()
    outTensor = TEST_TFMS(image=img)['image']

    return torch.unsqueeze(outTensor,0) 

def testImgAugAE(imgFrame):
    
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor()
        ])
    
    img = torch.squeeze(imgFrame).permute(2,0,1)
    im = transforms.ToPILImage()(img).convert("RGB")
    outTensor = transform(im)
    
    return torch.unsqueeze(outTensor,0)   

def testImgAugClsf(imgFrame):
    
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] )
        ])
    img = torch.squeeze(imgFrame).permute(2,0,1)
    im = transforms.ToPILImage()(img).convert("RGB")
    outTensor = transform(im)
     
    return torch.unsqueeze(outTensor,0)

#%% Extracting Features (Encoders)

def compute_features(frame, modelUNET, modelAE, modelCLSF, device):
    
    imageTensor = torch.from_numpy(frame)
    
    imageTensorUNET = testImgAugUNET(imageTensor).to(device)
    imageTensorAE = testImgAugAE(imageTensor).to(device)
    imageTensorCLSF = testImgAugClsf(imageTensor).to(device)
    
    modelUnetFeats = modelUNET.encoder.to(device)
    modelUnetFeats.eval()
    
    modelUnet = modelUNET.to(device)
    modelUnet.eval()
    
    modelAutoEncoder = modelAE.to(device)
    modelAutoEncoder.eval()
    
    modelClassifier = modelCLSF.to(device)
    modelClassifier.eval()
    
    
    seg_out = modelUnet(imageTensorUNET.float())
    mask = nn.Sigmoid()
    seg_out = mask(seg_out)
    seg_out = seg_out.cpu().detach().numpy()
    (frames,channels,rows,cols) = np.shape(seg_out)
    seg_outOrg = np.zeros((3,rows,cols))
    seg_outOrg[0,:,:]=np.squeeze(seg_out[:,0,:,:])
    seg_outOrg[1,:,:]=np.squeeze(seg_out[:,0,:,:])
    seg_outOrg[2,:,:]=np.squeeze(seg_out[:,0,:,:])
    
    seg_outOrg = np.moveaxis(seg_outOrg,0,-1)
    seg_outOrg = (seg_outOrg>0.5)
    seg_outOrg = np.round(seg_outOrg*255).astype(np.uint8)
    
    seg_Tensor = torch.from_numpy(seg_outOrg)
    seg_Tensor = testImgAugUNET(seg_Tensor).to(device)  
    
    feature_image = modelUnetFeats(imageTensorUNET.float())
    feature_seg = modelUnetFeats(seg_Tensor.float())
    
    feature_image = feature_image[5]    #Extract from last layer
    feature_image=feature_image.cpu().detach().numpy()    
    feature_image=np.mean(np.mean(feature_image,axis=3),axis=2)
    
    feature_seg = feature_seg[5]    #Extract from last layer
    feature_seg=feature_seg.cpu().detach().numpy()    
    feature_seg=np.mean(np.mean(feature_seg,axis=3),axis=2)
    
    featureCodeUNET = (0.7)*feature_image + (0.3)*feature_seg
    featureCodeUNET = featureCodeUNET.flatten()
    
    _, featureCodeAE = modelAutoEncoder(imageTensorAE.float())
    featureCodeAE = torch.squeeze(featureCodeAE)
    featureCodeAE = featureCodeAE.cpu().detach().numpy()
    
    probs = nn.Sigmoid()
    clsf, featsCLSF = modelClassifier(imageTensorCLSF.float())
    featureCodeCLSF = torch.squeeze(featsCLSF)
    featureCodeCLSF = featureCodeCLSF.cpu().detach().numpy()
    classOut = probs(clsf)
    classOut = torch.squeeze(classOut,0)
    classOut = classOut.cpu().detach().numpy()
    
    
    return featureCodeUNET, featureCodeAE, featureCodeCLSF, classOut, seg_out

#%% Decoder

class DecoderLSTM(nn.Module):
    #lstm
    def __init__(self, in_dim=512, hid_dim=256, num_layers=1, cell='lstm'):
        super(DecoderLSTM, self).__init__()
        
        self.Norm = nn.LayerNorm(512)
        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.combine = nn.Linear(512,1)
        self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hid_dim, 1)
        
    def forward(self, x_clsf, x_ae, x_seg):
        
        concated = torch.cat((x_clsf,x_ae,x_seg),0) #3,?,512
        
        #self attention and residual add + norm
        selfAtt = self.mha(concated, concated, concated)[0] #3,?,512
        addNorm = self.Norm(selfAtt+concated) #3,?,512
        
        attW = F.softmax(self.combine(addNorm), dim=0) #3,?,1
        attVector = torch.sum(addNorm*attW, dim=0, keepdim=True) #1,?,512
        
        attVector = self.Norm(attVector) #1,?,512
        h, _ = self.rnn(attVector)
        p = torch.sigmoid(self.fc(h))
        
        return p

#%% Loading Decoder

def LoadModelLSTM():
    decoder_name = r'lstm'
    in_dim = 512
    hid_dimLSTM = 256
    num_layersLSTM = 1
    filepath = r'modelWeights/decoder/model_ensembleEncoder_LSTMdecoder.pth.tar'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath,  map_location=torch.device(device))
    
    model = DecoderLSTM(in_dim=in_dim, hid_dim=hid_dimLSTM, 
                          num_layers=num_layersLSTM, cell=decoder_name)
    
    model.load_state_dict(checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad = False
    print("Loaded LSTM Trained Weights")    
    
    model.to(device)
    model.eval()
    
    return model

#%% Summmarization

class EncoderDecoder:
    def __init__(self, args, modelUNET, modelAE, modelCLSF, decoderLSTM, device):       
        # self.dataset = {}
        self.video_name=args.VIDEO_DIR.split('/')[-1]   
        self.video_path = args.VIDEO_DIR
        self.summary_path = args.SUMMARY_DIR        
        self.h5_path = args.h5Path
        self.video_list = []
        self._set_video_list(args.VIDEO_DIR)
        self.modelUNET = modelUNET
        self.modelAE = modelAE
        self.modelCLSF = modelCLSF
        self.decoderLSTM = decoderLSTM
        self.IMG_SIZE = args.IMG_SIZE
        self.device = device
        self.ClassFlag = args.ClassFlag
        self.SegFlag = args.SegFlag
        self.h5Flag = args.h5Flag
    

    def _set_video_list(self, video_path):
        #creating groups for each video
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = os.listdir(video_path)
            self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

    def encoderdecoder(self):
        #print(self.video_list)
        for video_idx, video_filename in enumerate(self.video_list):
            
            print(video_idx)
            
            if(not(os.path.exists(self.summary_path))):
                os.mkdir(self.summary_path)              
                        
            video_path = video_filename
            
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)
            
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
            featUNET,featAE,featCLSF = [],[],[]
            video_rewards_for_train = [] 
            segmentationOutput =[]
            videoArray=np.zeros((n_frames,int(video_capture.get(4)),int(video_capture.get(3)),1)) 
            videoArray = np.uint8(videoArray)
            
            for frame_idx in range(n_frames):
                success, frame = video_capture.read()
                R,C,F=np.shape(frame)
            
                if success:
                    videoArray[frame_idx,:,:,0] = frame[:,:,0] 
                    
                    frame = cv2.resize(frame,(self.IMG_SIZE,self.IMG_SIZE),interpolation = cv2.INTER_AREA)

                    frame_featUNET, frame_featAE, frame_featCLSF, classOut,seg_out = compute_features(frame, self.modelUNET, self.modelAE, self.modelCLSF, self.device)
                    
                    featUNET.append(frame_featUNET) 
                    featAE.append(frame_featAE) 
                    featCLSF.append(frame_featCLSF)
                    video_rewards_for_train.append(classOut)
                    segmentationOutput.append(seg_out)

                else:
                    break           
            video_capture.release() 
            
            seqClsf = torch.from_numpy(np.array(featCLSF)).unsqueeze(0)
            seqAE = torch.from_numpy(np.array(featAE)).unsqueeze(0)
            seqSeg = torch.from_numpy(np.array(featUNET)).unsqueeze(0)

            segmentationOutput = np.array(segmentationOutput).squeeze()
            
            seqClsf = seqClsf / torch.linalg.norm(seqClsf, dim=-1, keepdim=True)
            seqAE = seqAE / torch.linalg.norm(seqAE, dim=-1, keepdim=True)
            seqSeg = seqSeg / torch.linalg.norm(seqSeg, dim=-1, keepdim=True)
            
            seqClsf = seqClsf.to(self.device)
            seqAE = seqAE.to(self.device)
            seqSeg = seqSeg.to(self.device)
            
            probsLSTM = self.decoderLSTM(seqClsf, seqAE, seqSeg).squeeze(0).T
            
            ''' (2) T15S - Top 15% Segment Average Sampling'''
            segmentSize = 5
            Length = probsLSTM.size(1)
            segmentScoreLSTM = torch.zeros(1,Length-segmentSize+1)
            
            for i in range(Length-segmentSize+1):
                segmentScoreLSTM[0,i] = torch.mean(probsLSTM[0,i:i+segmentSize])
                
            topXpercent = int(0.15*(Length-segmentSize+1))
            
            segmentIdxLSTM = torch.topk(segmentScoreLSTM,topXpercent)[1]
            
            framesLSTM = torch.zeros(probsLSTM.shape)           
            
            for i in range(Length-segmentSize+1):
                if i in segmentIdxLSTM:
                    framesLSTM[0,i:i+segmentSize] = 1   
            
            framesLSTM = framesLSTM.T                    
            
            summaryFrames = np.array(framesLSTM)
            
            vid_writer_images = cv2.VideoWriter(self.summary_path+os.sep+str(video_filename),cv2.VideoWriter_fourcc(*'MPEG'),5,(C, R),)
            image=np.uint8(np.zeros([R,C,3]))
            
            for i in range(len(framesLSTM)):
                if (framesLSTM[i]==1):                    
                    frm = videoArray[i,:,:,0]                    
                    image=np.dstack([frm]*3)                   
                    if self.ClassFlag:
                        cv2.putText(image, str(video_rewards_for_train[i]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
                    if self.SegFlag:
                        tempFrame = np.uint8(255.0*cv2.resize(segmentationOutput[i], (C, R)))
                        mask = tempFrame>127
                        segFrame =np.uint8(255.0*np.dstack([mask]*3))
                        # bgr = 0,0,1 red mask
                        segFrame[:,:,0:2] = np.uint8(0.0*np.dstack([mask]*2))
                        image = cv2.addWeighted(image,1.0,segFrame,0.2,0)                       
                                                    
                    vid_writer_images.write(image)
            
            vid_writer_images.release()
            
            # Closes all the frames
            cv2.destroyAllWindows()
            
            if self.h5Flag:
            
                if(not(os.path.exists(self.h5_path))):
                    os.mkdir(self.h5_path)   
                    
                self.h5_file = h5py.File(self.h5_path+os.sep+video_filename[0:-4]+'.h5', 'w')
                            
                self.h5_file.create_group('video_{}'.format(video_idx+1))
                self.h5_file['video_{}'.format(video_idx+1)]['summaryFrames'] = list(summaryFrames)
                self.h5_file['video_{}'.format(video_idx+1)]['featUNET'] = list(featUNET)
                self.h5_file['video_{}'.format(video_idx+1)]['featAE'] = list(featAE)
                self.h5_file['video_{}'.format(video_idx+1)]['featCLSF'] = list(featCLSF)
                self.h5_file['video_{}'.format(video_idx+1)]['frameClsfScore'] = list(video_rewards_for_train)
                self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames# number of frames of video
                self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps #frames per second
                self.h5_file['video_{}'.format(video_idx+1)]['video_name'] = video_filename #name of input video
                self.h5_file['video_{}'.format(video_idx+1)]['video_full_name'] = video_path #name of input video
                
                self.h5_file.close()

#%%

def main():

    # Change working directory    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))    
    cwd = os.getcwd()
    parser = argparse.ArgumentParser("Code for summary generation")

    parser.add_argument('-s','--VIDEO_DIR', type=str, default=os.path.join(cwd,r'data'), help="path to input dataset (required)")
    parser.add_argument('--SUMMARY_DIR', type=str, default=os.path.join(cwd,r'summaryData'), help="path to output summary video")
    parser.add_argument('--h5Path', type=str, default=os.path.join(cwd,r'encFeatsH5'), help="path to encoded data")
    parser.add_argument('--IMG_SIZE', type=int, default=256, help="Image Size")
    parser.add_argument('--ClassFlag', type=bool, default=True, help="Whether Classification Label Needs to be Displayed or Not")
    parser.add_argument('--SegFlag', type=bool, default=True, help="Whether Segmentation OverLay Needs to be Displayed or Not")
    parser.add_argument('--h5Flag', type=bool, default=True, help="Whether features needs to saved as h5 file or Not")
    
    args = parser.parse_args()
    
    if(not(os.path.exists(args.SUMMARY_DIR))):
     os.mkdir(args.SUMMARY_DIR)
    
    ## Load Encoder Models
    modelUNET = smp.Unet(
            encoder_name='resnet34', # choose encoder
            encoder_depth = 5,
            encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
            in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            decoder_channels=(256, 128, 64, 32, 16)
        )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelUNET.load_state_dict(torch.load(r'modelWeights/preTrainedEncoders/bst_model128_fold4_0.8203.bin', map_location=torch.device(device)))
    modelUNET.eval()
    
    modelAE = loadModelFile(r'modelWeights/preTrainedEncoders/LungUSAE.pth')
    modelAE.eval()
    
    modelCLSF = loadModelFile(r'modelWeights/preTrainedEncoders/binaryLungStateClassifierCustomNet_32.pth')
    modelCLSF.eval()   
    
    ## Load Decoder Models
    decoderLSTM = LoadModelLSTM()
    decoderLSTM.eval()
    
    encdec = EncoderDecoder(args, modelUNET, modelAE, modelCLSF, decoderLSTM, device)
    encdec.encoderdecoder()    
    print("Summarization process complete!")

#%%

if __name__ == '__main__':
    main()

