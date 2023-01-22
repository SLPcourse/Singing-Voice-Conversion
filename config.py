import os

# Please configure the path of your downloaded datasets
dataset2path = {
    "Opencpop": "/mntnfs/lee_data1/zhangxueyao/dataset/Opencpop",
    "M4Singer": "/mntnfs/lee_data1/zhangxueyao/dataset/M4Singer",
}

# Please configure the path to save your data
data_path = "/mntnfs/lee_data1/zhangxueyao/Public/Singing-Voice-Conversion/preprocess"

# Wav files path
dataset2wavpath = {
    "Opencpop": os.path.join(dataset2path["Opencpop"], 'segments/wavs'),
    "M4Singer": dataset2path["M4Singer"],
}