# Singing Voice Conversion

This repository is to release the baselines of Singing Voice Conversion (SVC) task. To know more about SVC, you can read [this tutorial](https://www.zhangxueyao.com/data/SVC/tutorial.html). 

Now, it contains one model: WORLD-based SVC.

## Dataset

TBD

## WORLD-based SVC

[[Blog]](https://www.zhangxueyao.com/data/SVC/tutorial.html#Baseline) [[Demo]](https://www.zhangxueyao.com/data/SVC/tutorial.html#Demo)

![framework](https://www.zhangxueyao.com/data/SVC/data/framework.png)

This model is proposed by Xin Chen, et al:

> Xin Chen, et al. Singing Voice Conversion with Non-parallel Data. IEEE MIPR 2019.

As the figure above shows, there are two main modules of it, *Content Extractor* and *Singer-specific Synthesizer*. Given a source audio, firstly the *Content Extractor* is aim to extract content features (i.e., **singer independent features**) from the audio. Then, the *Singer-specific Synthesizer* is designed to inject the **singer dependent features** for the synthesis, so that the target audio can be able to capture the singer's characteristics.

We can utilize the following two stages to conduct **any-to-one** conversion:

1. **Acoustics Mapping Training** (Training Stage): This stage is to train the mapping from the textual content features (eg: PPG) to the target singer's acoustic features (eg: SP or MCEP). During this stage, we utilize the last layer encoder's output of [Whisper](https://github.com/openai/whisper) as the content features (which is 1024d). We use a 6-layer Transformer to train the mapping from whisper features to 40d MCEP features.
2. **Inference and Conversion** (Conversion Stage): Given any source singer's audio, firstly, extract its content features including F0, AP, and textual content features. Then, use the model of training stage to infer the converted acoustic features (SP or MCEP). Finally, given F0, AP, and the converted SP, we utilize WORLD as vocoder to synthesis the converted audio.

### Requirements

TBD

### Acoustic Mapping Training (Training Stage)

To write a overview

#### Input: Whisper Features

TBD

#### Output: MCEP Features

TBD

#### Training and Evaluation

TBD

### Inference and Conversion (Conversion Stage)

TBD

## License

TBD
