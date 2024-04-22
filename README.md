![Mamba for Sequence Classification](https://raw.githubusercontent.com/getorca/mamba_for_sequence_classification/428b7a3f8ca1b990875de4fbf52a5cc7ce22f3d1/docs/mamba_for_sequence_classification_sm.jpeg)

# Mamba for Sequence Classification

A huggingface Transformers compatibale implementation of Mamba for sequence classification. It add a Linear layer on top of the mamba model for classification. Offering a complete solution capitable with huggingface transformers. 

The code is based on the abandoned pull request <https://github.com/huggingface/transformers/pull/29552> from [Michael Schock](https://github.com/mjschock).

## Installation

currently clone this repo and run `pip install .` in your project. 


### Requirements:

Should be install as above or torch set up manually but...
- Torch
- Transformers
- causal-conv1d>=1.2.0 
- mamba-ssm

## Usage

The implementation should be compatable with everything huggingface has to offer for training and inference. See <https://huggingface.co/docs/transformers/en/index> and <https://huggingface.co/docs/transformers/en/tasks/sequence_classification> for more details on training with huggingface. 

A few examples realting to training mamba for sequence classification are available in <https://github.com/getorca/mamba_for_sequence_classification/blob/main/examples/>

*An import note for training:* make sure `use_cache=False` when loading a model for training with eval otherwise it will raise an error `'MambaCache' object has no attribute 'detach'`

When sharing a model on huggingface we recommend including the `hf_mamba_classification.py` file in the model repo. See <https://huggingface.co/docs/transformers/custom_models> for docs on building cusotm models. In the future maybe hf will pull this into the transformers repo, given enough examples and usage.

## Pretrained Models

Coming soon. 

## Citations

```
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

- <https://colab.research.google.com/drive/13EC5kbiZmtmFqBOsTW7j-A8JEVGEhvWg?usp=sharing#scrollTo=TycmnYlI36bR> -  a pytorch example of mamba for sequence classification in a notebook
