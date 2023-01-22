from whisper import whisper
import torch
import os
import json
import numpy as np
import sys

sys.path.append("../")
from config import data_path, dataset2wavpath


def whisper_encoder(audio_paths):
    batch = len(audio_paths)
    batch_mel = torch.zeros((batch, 80, 3000), dtype=torch.float, device=model.device)

    for i, audio_path in enumerate(audio_paths):
        # (48000,)
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        # (80, 3000)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        batch_mel[i] = mel

    with torch.no_grad():
        # (batch, 1500, 1024)
        features = model.embed_audio(batch_mel)

    return features.cpu().detach().numpy()


def get_whisper_features(dataset, dataset_type, batch_size=80):
    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)

    wave_dir = dataset2wavpath[dataset]
    with open(os.path.join(data_dir, "{}.json".format(dataset_type)), "r") as f:
        datasets = json.load(f)

    # Extract
    whisper_features = np.zeros((len(datasets), 1500, 1024), dtype=float)
    audio_paths = [
        os.path.join(wave_dir, "{}.wav".format(utt["Uid"])) for utt in datasets
    ]
    if dataset == "M4Singer":
        audio_paths = [os.path.join(wave_dir, utt["Path"]) for utt in datasets]

    start = 0
    end = 0
    while end <= len(audio_paths):
        start = end
        end = start + batch_size

        whisper_features[start:end] = whisper_encoder(audio_paths[start:end])
        print("{}/{}, Done.".format(min(len(audio_paths), end), len(audio_paths)))

    # Save
    output_dir = os.path.join(data_dir, "Whisper", "raw")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "{}.npy".format(dataset_type)), whisper_features)


if __name__ == "__main__":
    print("Loading Model...")
    model = whisper.load_model("medium")
    if torch.cuda.is_available():
        print("Using GPU...\n")
        model = model.cuda()
    else:
        print("Using CPU...\n")

    model = model.eval()

    get_whisper_features("Opencpop", 'train')
    get_whisper_features("Opencpop", 'test')
    get_whisper_features("M4Singer", "test")
