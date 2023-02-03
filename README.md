# Singing Voice Conversion

This repository is to release the baselines of Singing Voice Conversion (SVC) task. To know more about SVC, you can read [this tutorial](https://www.zhangxueyao.com/data/SVC/tutorial.html).

Now, it contains one model: WORLD-based SVC.

## Dataset

We adopt two public datasets, Opencpop [1] and M4Singer [2], to conduct **many-to-one** singing voice conversion. Specifically, we consider [Opencpop](https://wenet.org.cn/opencpop/) (which is a single singer dataset) as target singer and use [M4Singer](https://github.com/M4Singer/M4Singer) (which is a 20-singer dataset) as source singers.

You can download the datasets by onedrive: [[Opencpop]](https://cuhko365-my.sharepoint.com/:f:/g/personal/222042021_link_cuhk_edu_cn/EkA6sscoSVhOnArHjmPiujkBeRhZZjL31gSpxmzday0WHA?e=36RoKe) [[M4Singer]](https://cuhko365-my.sharepoint.com/:f:/g/personal/222042021_link_cuhk_edu_cn/EjhvMImgtcdKgDHmlReEGyMB_LEDHc8Z520n1VeyYxZ8Jw?e=ILi5k4).

> [1] Yu Wang, et al. Opencpop: A High-Quality Open Source Chinese Popular Song Corpus for Singing Voice Synthesis. InterSpeech 2022.
>
> [2] Lichao Zhang, et al. M4Singer: a Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus. NeurIPS 2022.

## WORLD-based SVC

[[Blog]](https://www.zhangxueyao.com/data/SVC/tutorial.html#Baseline) [[Demo]](https://www.zhangxueyao.com/data/SVC/tutorial.html#Demo)

![framework](https://www.zhangxueyao.com/data/SVC/data/framework.png)

This model is proposed by Xin Chen, et al:

> Xin Chen, et al. Singing Voice Conversion with Non-parallel Data. IEEE MIPR 2019.

As the figure above shows, there are two main modules of it, *Content Extractor* and *Singer-specific Synthesizer*. Given a source audio, firstly the *Content Extractor* is aim to extract content features (i.e., **singer independent features**) from the audio. Then, the *Singer-specific Synthesizer* is designed to inject the **singer dependent features** for the synthesis, so that the target audio can be able to capture the singer's characteristics.

We can utilize the following two stages to conduct **any-to-one** conversion:

1. **Acoustics Mapping Training** (Training Stage): This stage is to train the mapping from the textual content features (eg: PPG) to the target singer's acoustic features (eg: SP or MCEP).
2. **Inference and Conversion** (Conversion Stage): Given any source singer's audio, firstly, extract its content features including F0, AP, and textual content features. Then, use the model of training stage to infer the converted acoustic features (SP or MCEP). Finally, given F0, AP, and the converted SP, we utilize WORLD as vocoder to synthesis the converted audio.

### Requirements

```
pip install torch==1.13.1
pip install torchaudio==0.13.1
pip install pyworld==0.3.2
pip install diffsptk==0.5.0
pip install tqdm
```

### Dataset Preprocess

After you download the datasets, you need to modify the path configuration in `config.py`:

```python
# Please configure the path of your downloaded datasets
dataset2path = {"Opencpop": "[Your Opencpop path]",
    "M4Singer": "[Your M4Singer path]"}

# Please configure the root path to save your data and model
root_path = "[Root path for saving data and model]"
```

#### Opencpop

Transform the original Opencpop transcriptions to JSON format:

```
cd preprocess
python process_opencpop.py
```

#### M4Singer

Select some utterance samples that will be converted. We randomly sample 5 utterances for every singer:

```
cd preprocess
python process_m4singer.py
```

### Acoustic Mapping Training (Training Stage)

During training stage, we aim to train a mapping from textual content features to the target singer's acoustic features. In the implementation: (1) For output, we use 40d MCEP features as ground truth. (2) For input, we utilize the last layer encoder's output of [Whisper](https://github.com/openai/whisper) as the content features (which is 1024d). (3) For acoustic model, we adopt a 6-layer Transformer.

#### Output: MCEP Features

To obtain MCEP features, first we adopt [PyWORLD](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) to extract spectrum envelope (SP) features, and then transform SP to MCEP by using [diffsptk](https://github.com/sp-nitech/diffsptk):

```
cd preprocess
python extract_mcep.py
```

For hyparameters, we use 44100 Hz sampling rate and 10 ms frame shift.

#### Input: Whisper Features

To extract whisper features, we utilze the pretrained [multilingual models](https://github.com/openai/whisper#available-models-and-languages) of Whisper (specifically, `medium` model):

```
cd preprocess
git clone https://github.com/openai/whisper.git
python extract_whisper.py
```

#### Training and Evaluation

```
cd model
sh run_training.sh
```

### Inference and Conversion (Conversion Stage)

#### Inference

```
cd model
sh run_inference.sh
```

#### Conversion

```
cd model
sh run_converse.sh
```
