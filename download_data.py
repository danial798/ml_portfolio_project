import gdown
import os

# Dataset File ID from the PDF
file_id = '1e1S2rNn0_Bjv3XrUthw_M7MFoGuYcQE1'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'kidney_disease.csv'

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
else:
    print(f"{output} already exists.")
