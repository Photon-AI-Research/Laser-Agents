#!/bin/bash -l
#SBATCH -p hlab
#SBATCH -A hlab
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o ../logs/hostname_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=0

module load python/3.10.4
module load cuda/10.2

cd /home/bethke52/laser_data/scripts/
python3 /home/bethke52/laser_data/scripts/train_simple_agent_toy.py --savedir "/home/bethke52/laser_data/rl_tests/additive_rework/models/" --lr 1e-5 --n_episodes 20000000 --log_writer 1 --num_hidden 0 --episode_len 1 --log_every 2000 --batch_size 1024