#!/bin/bash
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1


module purge
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore5.4.0/lib64:$LD_LIBRARY_PATH

#srun python3 -u train_1.py

srun python3 "main.py" \
--meta_folder "VUAsequence/" \
--w2i_file "embeddings/w2i.json" \
--i2w_file "embeddings/i2w.json" \
--glove_filename "embeddings/embeddings.pickle" \
--elmo_weights "cached_elmo/weights_1024.hdf5" \
--elmo_options "cached_elmo/options_1024.json" \
--meta_train "train.csv" \
--meta_val "valid.csv" \
--meta_test "test.csv" \
--lr 0.005 --train_steps 3000 --dropout1 0.5 --dropout2 0.0 --dropout3 0.1 --num_layers 1 --hidden_size 100 --batch_size 32 --context_window 3 --model_type "VA_LSTM" --repetitions 10 --output_name "VA_LSTM_CW_3" --use_elmo 1

