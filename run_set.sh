for i in 10 20 30 40 50 
do
	for j in  0.1 0.5 1.0 1.5
	do
		CUDA_VISIBLE_DEVICES=5 python set_MI_classification.py --cnn-path-dir model/cnn/celeba/ --exp_name temp --hidden-dim ${i} --lr ${j}
	done
done