from tqdm import tqdm
import os
import numpy as np
import pickle
import librosa
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/zhangxueyao/SVC/")
from preprocess import extract_sp, extract_mcep
from inference import get_uids, save_audio

data_dir = "/mntnfs/lee_data1/zhangxueyao/SVC/preprocess"


def converse_base_f0(target_singer_f0_file, ratio=0.25):
    # Loading target singer's F0 statistics
    with open(target_singer_f0_file, "rb") as f:
        mean, total_f0 = pickle.load(f)

    total_f0.sort()
    base = total_f0[int(len(total_f0) * ratio)]
    print("Target: mean = {}, ratio = {}, base = {}".format(mean, ratio, base))
    return base


def get_pred_audios(
    model_file,
    source_dataset,
    dataset_type,
    target_singer_f0_file=None,
    conversion_dir=None,
):
    if source_dataset == "Opencpop":
        wave_dir = "/mntnfs/lee_data1/zhangxueyao/dataset/Opencpop/segments/wavs"
    if source_dataset == "OpencpopBeta":
        wave_dir = "/mntnfs/lee_data1/zhangxueyao/dataset/OpencpopBeta/utt_wavs"
    if source_dataset == "M4Singer":
        wave_dir = "/mntnfs/lee_data1/zhangxueyao/dataset/M4Singer"

    paths = model_file.split("/")
    epoch = paths[-1].split(".")[0]

    if source_dataset == "OpencpopBeta":
        pred_dir = "/".join(paths[:-1])
    else:
        pred_dir = conversion_dir

    output_dir = os.path.join(pred_dir, "{}_{}".format(dataset_type, epoch))
    if os.path.exists(output_dir):
        os.system("rm -r {}".format(output_dir))
    os.makedirs(output_dir)

    pred_file = os.path.join(pred_dir, "{}_{}.npy".format(dataset_type, epoch))
    # (dataset_sz, seq_len, MCEP_dim)
    pred = np.load(pred_file)
    print("File: {},\n Shape: {}\n".format(pred_file, pred.shape))

    # Loading base f0 (the lowest pitch in an utterance)
    ratio = 0.5
    target_base_f0 = converse_base_f0(target_singer_f0_file, ratio=ratio)

    # Predict
    uids = get_uids(source_dataset, dataset_type)
    if source_dataset == "M4Singer":
        uids, upaths = uids

    for i, uid in enumerate(tqdm(uids)):
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))
        if source_dataset == "M4Singer":
            wave_file = os.path.join(wave_dir, upaths[i])

        f0, sp, ap, fs = extract_sp.extract_world_features(wave_file)

        frame_len = min(len(f0), len(pred[i]))
        f0 = f0[:frame_len]
        sp = sp[:frame_len]
        ap = ap[:frame_len]

        # Get sp_pred
        mcep = pred[i][:frame_len]
        sp_pred = extract_mcep.mgc2SP(mcep)
        assert sp.shape == sp_pred.shape

        # Get transposed f0
        source_base_f0 = sorted([f for f in f0 if f != 0])
        source_base_f0 = source_base_f0[int(len(source_base_f0) * ratio)]
        f0_trans = f0 * (target_base_f0 / source_base_f0)
        print("File: {}, mapping = {}".format(uid, target_base_f0 / source_base_f0))

        # Synthesis by WORLD
        y_gt = extract_sp.world_synthesis(f0, sp, ap, fs)
        y_pred = extract_sp.world_synthesis(f0, sp_pred, ap, fs)
        y_pred_trans = extract_sp.world_synthesis(f0_trans, sp_pred, ap, fs)

        # save gt
        gt_file = os.path.join(output_dir, "{}.wav".format(uid))
        os.system("cp {} {}".format(wave_file, gt_file))
        # save WORLD synthesis gt
        world_gt_file = os.path.join(output_dir, "{}_world.wav".format(uid))
        save_audio(world_gt_file, y_gt, fs)
        # save WORLD synthesis pred
        world_pred_file = os.path.join(output_dir, "{}_pred.wav".format(uid))
        save_audio(world_pred_file, y_pred, fs)
        # save WORLD synthesis pred (f0 transposed)
        world_pred_trans_file = os.path.join(
            output_dir, "{}_pred_trans.wav".format(uid)
        )
        save_audio(world_pred_trans_file, y_pred_trans, fs)


if __name__ == "__main__":
    model_file = "/mntnfs/lee_data1/zhangxueyao/SVC/model/ckpts/Opencpop/whisper_Transformer_lr_0.0001/118.pt"
    # get_pred_audios(model_file, "OpencpopBeta", "test")
    get_pred_audios(
        model_file,
        source_dataset="M4Singer",
        dataset_type="test",
        target_singer_f0_file="/mntnfs/lee_data1/zhangxueyao/SVC/preprocess/Opencpop/F0/test_f0.pkl",
        conversion_dir="/mntnfs/lee_data1/zhangxueyao/SVC/model/ckpts/M4Singer/whisper_Transformer_eval_conversion",
    )
