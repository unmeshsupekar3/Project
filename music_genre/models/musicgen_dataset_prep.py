import os
import librosa
import math
import json


#constants or globals
DATASET_PATH = "C://Users//unmes//Downloads//musicgen//Data//genres_original"
JSON_PATH = "C://Users//unmes//Downloads//musicgen//Data//data.json"
SAMPLE_RATE = 22050 
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION



def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    """Extracts MFCCs from music dataset and 
    saves them into a json file along with the
    genre labels.
    
    here,
    
      dataset_path (str): Path to dataset
      
      json_path (str): Path to json file 
                      used to save MFCCs
                      
      num_mfcc (int): Number of coefficients 
                      to extract
                      
      n_fft (int): Interval we consider to
                      apply FFT. Measured in
                      number of samples
                      
      hop_length (int): Sliding window for 
                          FFT. Measured in number
                          of samples
                          
      num_segments (int): Number of segments we
                          want to divide sample tracks 
                          
      """
      
    data = {
     "mapping": [],
     "labels": [],
     "mfcc": []
 }    
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
      # the expected num of vect value may have float value  
      # we need 1.3 as 2 hence we use ceil 
      
      
      
      
      
      #looping through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
          
          
          # ensure we are not at the root level or dataset level
          # because we have to go through all the folders of datasets
          # we have to be at genre level while we are looping
          
          
          
          if dirpath is not dataset_path:
              
              #save the semantic labels as classical blues etc in mappings
              dirpath_components= dirpath.split("/") 
              #genre/blues will be saved as ["genre','blues.]
              semantic_label= dirpath_components[-1]
              data["mapping"].append(semantic_label)
              
              
              
              print("\n Processing: {}".format(semantic_label))
              
              
              
              #process file for a specific genre
              for f in filenames:
                  
                  #joining dirpath and filename from loop to load audio file
                  file_path = os.path.join(dirpath,f)
                  signal, sr= librosa.load(file_path, sr=SAMPLE_RATE)
                  
                  
                  
                  '''we need to process data as segments while we extract mfcc 
                  for each segment
                  hence process segments extracting mfcc and storing data'''
                  
                  
                  for s in range(num_segments):
                      start_sample = num_samples_per_segment * s 
                      # here the start sample will be at zero s=0 ->> 0
                      # s is current segment we are in
                      finish_sample = start_sample + num_samples_per_segment 
                      #here the finish sample will be -> num_samples_er_segment 
                      
                      
                  
                      
                      
                      mfcc = librosa.feature.mfcc(y=signal[start_sample : finish_sample], 
                                                  sr = sr, 
                                                  n_fft = n_fft,
                                                  n_mfcc = n_mfcc,
                                                  hop_length = hop_length)
                      mfcc = mfcc.T
                      
                  
                      
                      #store mfcc for segment if it has expected length
                      if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                          data["mfcc"].append(mfcc.tolist())
                          data["labels"].append(i-1)
                          
                          
                          
                          print("{}, segment:{}".format(file_path, s+1))
    
    
    
    with open(json_path,"w") as fp:
        json.dump(data, fp, indent=4)
     
        
     
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH,JSON_PATH,num_segments=10)
                          
                  
          
          
      
      
      
      
      
      

