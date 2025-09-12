import requests
import os

url = "http://127.0.0.1:8000/predict"

files = {
    "image_zip": (
        os.path.basename("data/mc/super_small_image.zip"),
        open("data/mc/super_small_image.zip", "rb"),
        "application/x-zip-compressed"
    ),
    "labels_csv": (
        os.path.basename("data/mc/super_small_data.csv"),
        open("data/mc/super_small_data.csv", "rb"),
        "text/csv"
    ),
}

headers = {"accept": "application/json"}

response = requests.post(url, files=files, headers=headers)

try:
    print(response.json())
except Exception:
    print(response.text)

# ### INPUT
# curl -X 'POST' \
#   'http://47.186.63.142:52901/predict' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: multipart/form-data' \
#   -F 'image_zip=@super_small_image.zip;type=application/x-zip-compressed' \
#   -F 'labels_csv=@super_small_data.csv;type=text/csv'

# ### OUTPUT
# {
#     "status": "success",
#     "infer_time": 0.5484125036746264,
#     "predictions": [
#         {
#             "key": 0,
#             "class": 0,
#             "confidence": 0.94629967212677
#         },
#         {
#             "key": 1,
#             "class": 1,
#             "confidence": 0.6092917323112488
#         }
#     ]
# }
