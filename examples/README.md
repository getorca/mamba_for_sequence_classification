# Examples

## `train_mamba_imdb.py` 

- an implementation of training a sentiment classifier on the IMDB dataset using Mamba. The original tutorial is here <https://huggingface.co/docs/transformers/en/tasks/sequence_classification>.
- the script can be run with ddp for example: `OMP_NUM_THREADS=4 WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port=1234 train_mamba_imdb.py``