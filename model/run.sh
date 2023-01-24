# # ===================== Training and Inferring =====================

# ********* Transformer ********* 

# # PPG
# python -u main.py --debug False \
# --dataset 'OpencpopBeta' --model 'Transformer' \
# --lr 1e-4 --batch_size 32 --epochs 500

# # Whisper 
# python -u main.py --debug False \
# --dataset 'Opencpop' --input 'whisper' --model 'Transformer' \
# --lr 1e-4 --batch_size 32 --epochs 500

# ********* MLP ********* 

# # PPG
# python -u main.py --debug False \
# --dataset 'OpencpopBeta' --model 'MLP' \
# --lr 1e-3 --batch_size 32 --epochs 500

# # Y+noise -> Y
# python -u main.py --debug False --noise_debug True \
# --dataset 'OpencpopBeta' --model 'MLP' \
# --lr 1e-3 --batch_size 32 --epochs 500

# # Whisper
# python -u main.py --debug False \
# --dataset 'OpencpopBeta' --input 'whisper' --model 'MLP' \
# --lr 1e-1 --batch_size 32 --epochs 500

# # PPG + Whisper features
# python -u main.py --debug False \
# --dataset 'OpencpopBeta' --input 'ppg_whisper' --model 'MLP' \
# --lr 1e-3 --batch_size 32 --epochs 500

# ===================== Only Eval on OpencpopBeta =====================
# python -u main.py --debug False --evaluate True \
# --dataset 'OpencpopBeta' --model 'Transformer' \
# --resume '/mntnfs/lee_data1/zhangxueyao/SVC/model/ckpts/OpencpopBeta/Transformer_0.0001/6.pt'

# ===================== Conversion from Opencpop =====================
# python -u main.py --debug False --evaluate True \
# --dataset 'Opencpop' --converse True --model 'Transformer' --input 'whisper' \
# --resume '/mntnfs/lee_data1/zhangxueyao/SVC/model/ckpts/OpencpopBeta/whisper_Transformer_lr_0.0001/79.pt'

# ===================== Conversion from M4Singer =====================
python -u main.py --debug False --evaluate True \
--dataset 'M4Singer' --converse True --model 'Transformer' --input 'whisper' \
--resume '/mntnfs/lee_data1/zhangxueyao/SVC/model/ckpts/Opencpop/whisper_Transformer_lr_0.0001/118.pt'