#CUDA_VISIBLE_DEVICES=0 python main.py --exp_name temp_2 --gan-path model/gan/GS_WGAN_epsinf/netGS.pth --cnn-path model/cnn/epsinf/model.pt

for i in 1 10 100 
do
	for j in 0.015 0.05 0.1
	do
		CUDA_VISIBLE_DEVICES=5 python main_celeba.py --exp_name exp_l2_${i}_lr_${j}_l3_1.0 -l2 ${i} -l3 1.0 -lr ${j} --gan-path model/gan/celeba/netGS_20000.pth --cnn-path model/cnn/celeba/net_epoch_10.pth
	done
done




