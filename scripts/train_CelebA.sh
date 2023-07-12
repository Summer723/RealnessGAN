#!/bin/bash
#SBATCH --output=%N-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:v100l:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=50:00:00


nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 >> gpu_usage.log &
smi_pid=$!

# Start monitoring CPU usage with top
top -b -d 1 -u $USER >> cpu_usage.log &
top_pid=$!

# Train DCGAN using your preferred training script
# tensorboard --logdir runs --host 0.0.0.0 
# put data to the node would make training significantly faster
source /home/summer23/scratch/ENV/bin/activate
cd $SLURM_TMPDIR
tar -xf /home/summer23/scratch/HD-CelebA-Cropper/data/aligned/celeba.tar
echo Uncompress Done

cd /home/summer23/scratch/RealnessGAN #directory to RealnessGAN
python3 train.py --lr_D 0.0002 --lr_G 0.0002 \
		--beta1 0.5 --beta2 .999 \
		--gen_every 10000 --print_every 10000 \
		--G_h_size 32 --D_h_size 32 \
		--gen_extra_images 5000 \
		--input_folder $SLURM_TMPDIR \
		--image_size 256 \
		--total_iters 500000 \
		--seed 999 \
		--num_workers 8 \
        	--G_updates  1 \
       	 	--D_updates  1 \
       		--effective_batch_size   32 \
        	--batch_size   32 \
        	--n_gpu         2  \
        	--positive_skew   1.0 \
        	--negative_skew  -1.0 \
        	--num_outcomes     51 \
        	--use_adaptive_reparam   False \
        	--relativisticG         True \
		--output_folder "/home/summer23/scratch/output"   --extra_folder "/home/summer23/scratch/extra" 
# Kill the monitoring processes
kill $smi_pid
kill $top_pid
