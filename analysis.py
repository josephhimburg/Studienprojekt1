import os
import cv2
from deepface import DeepFace
import pandas as pd

input_folder = 'test'
png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

results_list = []
csv_file = 'results.csv'

for png_file in png_files:
    img_path = os.path.join(input_folder, png_file)
    img = cv2.imread(img_path)
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    
    if isinstance(results, list):
        result = results[0]
    else:
        result = results
    
    result['image_name'] = png_file  

    results_list.append(result)

result_df = pd.DataFrame(results_list)
result_df.to_csv(csv_file, index=False)

print(result_df)