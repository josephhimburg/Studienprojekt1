import pandas as pd 

df = pd.read_csv('results.csv') 

confidence_min90 = df[df['face_confidence'] >= 0.9] 
dominant_emotion_counts_min90 = confidence_min90['dominant_emotion'].value_counts() 

print(dominant_emotion_counts_min90) 


