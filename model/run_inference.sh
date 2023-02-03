# ===================== Inference for M4Singer =====================
# Note:
# --resume:
#   <root_path>/model/ckpts/<training_dataset>/<expriement_name>/<epoch>.pt

python -u main.py --debug False --evaluate True \
--dataset 'M4Singer' --converse True --model 'Transformer' \
--resume '/mntnfs/lee_data1/zhangxueyao/Public/Singing-Voice-Conversion/model/ckpts/Opencpop/Transformer_lr_0.0001/96.pt'