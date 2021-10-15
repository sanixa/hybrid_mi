for i in 10000 15000 20000 50000 
do
	for j in  0.9 1.0 1.1 1.5
	do
		for k in 10000 15000 20000 50000
		do
			for l in  1.1
			do
                echo ${k} ${l}
				CUDA_VISIBLE_DEVICES=1 python tools/eval_roc.py --attack_type "wb" -ldir result/exp_l2_${i}_lr_${j}_l3_0.5 -rdir result/exp_l2_${k}_lr_${l}_l3_1.0
			done
		done
	done
done