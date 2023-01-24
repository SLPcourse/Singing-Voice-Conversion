from tqdm import tqdm
import os
import numpy as np
import json
import torchaudio
import torch
import random
import sys
import librosa
import matplotlib.pyplot as plt
from torchaudio.transforms import MelSpectrogram

sys.path.append("/home/zhangxueyao/SVC/")
from preprocess import extract_sp, extract_mcep

data_dir = "/mntnfs/lee_data1/zhangxueyao/SVC/preprocess"


def get_uids(dataset, dataset_type):
    dataset_file = os.path.join(data_dir, dataset, "{}.json".format(dataset_type))
    with open(dataset_file, "r") as f:
        utterances = json.load(f)

    if dataset == "M4Singer":
        uids = ["{}_{}_{}".format(u["Singer"], u["Song"], u["Uid"]) for u in utterances]
        upaths = [u['Path'] for u in utterances]
        return uids, upaths
    else:
        return [u["Uid"] for u in utterances]


def save_audio(path, waveform, fs):
    # (frames,) numpy, float64 ->
    # 16 bit music, 8 bit music
    waveform = torch.as_tensor(waveform, dtype=torch.int16)
    if len(waveform.size()) == 1:
        waveform = waveform[None, :]
    # print("HERE: waveform", waveform.shape, waveform.dtype, waveform)
    torchaudio.save(path, waveform, fs)


def save_pred_audios_in_training(
    pred, args, loss, dataset_type, sample_sz=1, random_sampling=False
):
    dataset = args.dataset
    if dataset == "OpencpopBeta":
        wave_dir = "/mntnfs/lee_data1/zhangxueyao/dataset/OpencpopBeta/utt_wavs"
    else:
        assert dataset == "Opencpop"
        wave_dir = "/mntnfs/lee_data1/zhangxueyao/dataset/Opencpop/segments/wavs"

    output_dir = os.path.join(args.save, "{}".format(dataset_type))
    os.makedirs(output_dir, exist_ok=True)

    # Predict
    uids = get_uids(dataset, dataset_type)
    if random_sampling:
        sample_uids = random.sample(uids, sample_sz)
    else:
        sample_uids = uids[:sample_sz]

    # MelSpectrogram
    default_fs = 44100
    MelSpectrogramTransform = MelSpectrogram(sample_rate=default_fs)

    for uid in sample_uids:
        i = uids.index(uid)
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))
        f0, sp, ap, fs = extract_sp.extract_world_features(wave_file)
        frame_len = len(f0)
        assert fs == default_fs

        mcep = pred[i][:frame_len]
        sp_pred = extract_mcep.mgc2SP(mcep)
        assert sp.shape == sp_pred.shape

        y_gt = extract_sp.world_synthesis(f0, sp, ap, fs)
        y_pred = extract_sp.world_synthesis(f0, sp_pred, ap, fs)

        # save gt
        gt_file = os.path.join(output_dir, "{}_gt.wav".format(uid))
        os.system("cp {} {}".format(wave_file, gt_file))
        # save WORLD synthesis gt
        world_gt_file = os.path.join(output_dir, "{}_gt_world.wav".format(uid))
        save_audio(world_gt_file, y_gt, fs)
        # save WORLD synthesis pred
        world_pred_file = os.path.join(
            output_dir,
            "{}_pred_{}_loss{:.5f}.wav".format(uid, args.current_epoch, loss),
        )
        save_audio(world_pred_file, y_pred, fs)

        # save melspectrogram
        gt_file = gt_file.replace(".wav", ".jpg")
        waveform, _ = torchaudio.load(wave_file)
        save_melspectrogram(waveform, MelSpectrogramTransform, gt_file)

        world_gt_file = world_gt_file.replace(".wav", ".jpg")
        save_melspectrogram(y_gt, MelSpectrogramTransform, world_gt_file)

        world_pred_file = world_pred_file.replace(".wav", ".jpg")
        save_melspectrogram(y_pred, MelSpectrogramTransform, world_pred_file)


def plot_spectrogram(specgram, title="MelSpectrogram", ylabel="mel freq"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def save_melspectrogram(waveform, transform, path):
    """
    # TODO
    """
    # waveform = torch.as_tensor(waveform)
    # mel_specgram = transform(waveform)  # (channel, n_mels, time)
    pass
