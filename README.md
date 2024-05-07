**Automated detectors of Antarctic baleen whale sounds** 

This repositiry propose a simple automated detectors of Antarctic baleen whale sounds.
Model archytecture and weights can be found in the folder model. New models with new species will be added soon.   

A python notebook ```tuto.ipynb``` is proposed to apply model on two 6-hour audio files recorded in the Indian Ocean contaning annotated Blue whale FM (AKA D-calls). 

All dependencies can be found in ```requirements.txt```

• [mscl_cnn14_pt](#mscl_cnn14_pt) 

### mscl_cnn14_pt

This model has been trained using the online dataset proposed by Miller at al. (2021) [1]. It has been trained to detect seven types of baleen whale vocalizations : Antarctic blue whale unit A, Antarctic blue whale unit AB, Antarctic blue whale z-call, Blue whale FM (AKA D-calls), Fin whale 20 Hz pulse, Fin whale 20 Hz pulse with energy at higher frequencies (e.g. 89 or 99 Hz components) and Fin whale FM calls (AKA ‘high frequency’ downsweep; AKA 40 Hz pulse). More information can be found on the original paper. 

**Please, contact me to get the weights of the encoder : gabriel.dubus@dalembert.upmc.fr**



[1] Miller, B.S., The IWC-SORP/SOOS Acoustic Trends Working Group, Miller, B.S., Stafford, K.M., Van Opzeeland, I., Harris, D., Samaran, F., Širović, A., Buchan, S., Findlay, K., Balcazar, N., Nieukirk, S., Leroy, E.C., Aulich, M., Shabangu, F.W., Dziak, R.P., Lee, W.S., Hong, J.K., 2021. An open access dataset for developing automated detectors of Antarctic baleen whale sounds and performance evaluation of two commonly used detectors. Sci Rep 11, 806. https://doi.org/10.1038/s41598-020-78995-8