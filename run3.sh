#CUDA_VISIBLE_DEVICES=0 python main.py --exp_name temp_2 --gan-path model/gan/GS_WGAN_epsinf/netGS.pth --cnn-path model/cnn/epsinf/model.pt

for i in 10000 15000 20000 50000 
do
	for j in  0.9 1.0 1.1 1.5
	do
		CUDA_VISIBLE_DEVICES=2 python main_celeba.py --exp_name exp_l2_${i}_lr_${j}_l3_1.0_60k -l2 ${i} -l3 1.0 -lr ${j} --gan-path model/gan/celeba_60k/netGS_100000.pth --cnn-path model/cnn/celeba/net_epoch_10_60k.pth
	done
done
