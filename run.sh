#CUDA_VISIBLE_DEVICES=0 python main.py --exp_name temp_2 --gan-path model/gan/GS_WGAN_epsinf/netGS.pth --cnn-path model/cnn/epsinf/model.pt
if [[ "$1" == "cifar" ]]
then
	for i in 0.01 0.04 0.08 0.1 0.2
	do
		for j in 0.01 0.015 0.05 0.1
		do
			CUDA_VISIBLE_DEVICES=0 python main.py --exp_name exp_l2_${i}_lr_${j}_l3_0.2 -l2 ${i} -l3 0.2 -lr ${j} --gan-path model/gan/GS_WGAN_epsinf/netGS.pth --cnn-path model/cnn/epsinf/model.pt
		done
	done
else
	for i in 0.01 0.04 0.08 0.1 0.2
	do
		for j in 0.01 0.015 0.05 0.1
		do
			CUDA_VISIBLE_DEVICES=0 python main.py --exp_name exp_l2_${i}_lr_${j}_l3_0.2 -l2 ${i} -l3 0.2 -lr ${j} --dataset_name mnist --gan-path model/gan/GS_WGAN_epsinf/netGS.pth --cnn-path model/cnn/epsinf/model.pt
		done
	done
fi

