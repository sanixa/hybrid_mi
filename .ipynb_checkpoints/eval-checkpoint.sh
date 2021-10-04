for i in 0.05 0.1 0.2 0.5
do
    for j in 0.015 0.05 0.1 0.2 0.5
    do
        CUDA_VISIBLE_DEVICES=0 python tools/eval_roc.py --attack_type "wb" -ldir result/exp_l2_${i}_lr_${j}_l3_0.5
    done
done

for i in 0.05 0.1 0.2 0.5
do
    for j in 0.015 0.05 0.1 0.2 0.5
    do
        CUDA_VISIBLE_DEVICES=0 python tools/eval_roc.py --attack_type "wb" -ldir result/exp_l2_${i}_lr_${j}_l3_1.0
    done
done
