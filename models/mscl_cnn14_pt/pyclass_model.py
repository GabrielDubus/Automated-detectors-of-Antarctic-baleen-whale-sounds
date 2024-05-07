# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:27:57 2024

@author: gabri
"""

from transform_functions import transform, gen_spectro
from models import CNN14, LinearClassifier
import numpy as np
import os
import torch
from torchvision import transforms

class Model():
    def __init__(self, base_path):

        self.model_folder = 'mscl_cnn14_pt'
        self.parameters_Model = np.load(base_path + os.sep + self.model_folder + os.sep + 'hyperparameters.npz', allow_pickle=True)['parameters'].item()
        #Load Parameters for pre-processing    
        self.nfft_tab = self.parameters_Model['nfft']
        self.window_size_tab = self.parameters_Model['window_size']
        self.overlap_tab = self.parameters_Model['overlap']
        self.dynamic_min_tab = self.parameters_Model['dynamic_min']
        self.dynamic_max_tab = self.parameters_Model['dynamic_max']
        self.input_data_format = self.parameters_Model['input_data_format']
        self.label_to_detect  = self.parameters_Model['label_to_detect']
        self.sample_rate = self.parameters_Model['sample_rate'][0]
        self.LenghtFile = 50
        
        self.parameters_trans = {'dynamic_min':self.dynamic_min_tab, 'dynamic_max':self.dynamic_max_tab, 'index_dataset':0}
        
        #Load Parameters for model  
        #%% DEVICE 
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.sequence_length = self.parameters_Model['sequence_length']
        self.DEFAULT_NUM_CLASSES = 7
        self.DEFAULT_OUT_DIM = 512
        self.PATH_TO_WEIGHTS = 'lala' #os.path.join(codes_path, weightspath, 'Cnn6_mAP=0.343.pth')
        self.architecture_encoder = 'cnn14'

        self.trans = transform

        self.encoder = CNN14(num_classes=self.DEFAULT_NUM_CLASSES, do_dropout=True, embed_only=True, device=self.device)
        self.classifier = LinearClassifier(name=self.architecture_encoder, num_classes=self.DEFAULT_NUM_CLASSES, device=self.device)

        self.encoder.load_state_dict(torch.load(base_path + os.sep + self.model_folder + os.sep + 'weights' + os.sep + self.model_folder + '_encoder.pt'))
        self.classifier.load_state_dict(torch.load(base_path + os.sep + self.model_folder + os.sep + 'weights' + os.sep + self.model_folder + '_classifier.pt'))    

        self.encoder.eval()
        self.classifier.eval()
        
        print('__________________')
        print('REQUIERMENT : ')
        print('Use model with : output = apply_model(audio)')
        print('audio (array), is a 50-second audio segment with a sampling rate at 250Hz')
        print('output is a list of float with detection results [0,1] for the following calls : ')
        print("['Bp20Hz' 'Bp20Plus' 'BpDS' 'BmA' 'BmB' 'BmZ' 'BmD']")
        print('__________________')
        
            
    def apply_model(self, audio):
        spectro, sxx = gen_spectro(audio, self.sample_rate, self.window_size_tab[0], self.overlap_tab[0], self.nfft_tab[0])
        input_data = self.trans(spectro, self.parameters_trans)
        
        #to device
        input_data = input_data.to(self.device)
        #apply model
        lin = self.encoder(input_data[None, None,:].float())
        outputs_batch = self.classifier(lin)
        return outputs_batch.cpu().detach().numpy()
        