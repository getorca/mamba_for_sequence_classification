# Examples

## Finacial Phrase Bank Ray PBT

- this example uses ray and ray tune to hunt for hyperparms with population based training (PBT) on the financial phrase bank dataset. Ray alone does significantly slow down training over DDP on a single node, but PBT and hyperparam tuning significantly improves the model. Note hyperparam tuning also relies on training the model multiple times.
- run `python train_finacial_phrasebank_ray_pbt.py` to start the training.

## `train_mamba_imdb.py` 

- an implementation of training a sentiment classifier on the IMDB dataset using Mamba. The original tutorial is here <https://huggingface.co/docs/transformers/en/tasks/sequence_classification>.
- the script can be run with ddp for example: `OMP_NUM_THREADS=4 WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port=1234 train_mamba_imdb.py``


