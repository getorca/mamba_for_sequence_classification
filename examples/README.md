# Examples

## Finacial Phrase Bank Ray PBT

- this example uses ray and ray tune to hunt for hyperparms with population based bandets (PB2) on the financial phrase bank dataset. Ray alone does significantly slow down training over DDP on a single node, but PBT and hyperparam can significantly improve the model and simplyfy the process of finding the optimal hyperparms. Note hyperparam tuning relies on training the model multiple times.
- install ray and ray[tune] with `pip install ray[tune]`
- run `python train_finacial_phrasebank_ray_pbt.py` to start the training.
- for inference check `inference_example.ipynb`
- the pretrained model scoring 0.84 on accuracy is available on HF @ <https://huggingface.co/winddude/mamba_finacial_phrasebank_sentiment>

## `train_mamba_imdb.py` 

- an implementation of training a sentiment classifier on the IMDB dataset using Mamba. The original tutorial is here <https://huggingface.co/docs/transformers/en/tasks/sequence_classification>.
- the script can be run with ddp for example: `OMP_NUM_THREADS=4 WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port=1234 train_mamba_imdb.py``


