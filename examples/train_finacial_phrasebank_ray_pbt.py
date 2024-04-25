import os
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
import torch
from hf_mamba_classification import MambaForSequenceClassification
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset

import ray
from ray.train.torch import TorchTrainer
from ray.train import (
    ScalingConfig,
    ScalingConfig, 
    CheckpointConfig
)
from ray.train.huggingface.transformers import (
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune, train
from ray.tune import Tuner
from ray.tune.schedulers import PopulationBasedTraining


model_path = 'state-spaces/mamba-2.8b-hf'
# model_path = 'state-spaces/mamba-130m-hf'


def training_func(config):
    
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

    # model 
    model = MambaForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id,
        use_cache=False  # This needs to be passed when using eval and training Mamba for sequence classification otherwise it will raise an error
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True)
    
    # Load Datasets
    # we'll use my `https://huggingface.co/datasets/winddude/finacial_pharsebank_66agree_split` 
    # It's saved with splits and takes only options where 66% agree so we get only atrong predictions and the same split each run.
    # ========================
    imdb = load_dataset("winddude/finacial_pharsebank_66agree_split")

    def preprocess_function(examples):
        return tokenizer(examples["sentence"], max_length=512, truncation=True)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    tokenized_imdb = tokenized_imdb.remove_columns('attention_mask')
    train_dataset = tokenized_imdb["train"]
    eval_dataset = tokenized_imdb["test"]

    # Evaluation Metrics
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Hugging Face Trainer
    # ====================    
    
    training_args = TrainingArguments(
        output_dir="/files/training/mamba_finacial_phrasebank_sentiment",
        learning_rate=config.get("learning_rate", 5e-5),
        per_device_train_batch_size=2,  # 4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4, # 16,
        num_train_epochs=config.get("epochs", 3),
        weight_decay=config.get("weight_decay", 0.01),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        lr_scheduler_type="cosine",
        # optim='paged_adamw_8bit',
        push_to_hub=False,        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,       
    )

    # [3] Inject Ray Train Report Callback
    # ====================================
    trainer.add_callback(RayTrainReportCallback())  # `RayTrainReportCallback` binds the latest metrics and checkpoints together.

    # [4] Prepare your trainer
    # ========================
    trainer = prepare_trainer(trainer)
    trainer.train()

    
# see: https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
scaling_config = ScalingConfig(
    num_workers=1, 
    use_gpu=True,
    resources_per_worker={
        "CPU": 4,
        "GPU": 1,
    },
)

# see: https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html#ray.train.torch.TorchTrainer
trainer = TorchTrainer(
    training_func,
    scaling_config=scaling_config
)


# Prepare the population based trainer in ray
# guide -> https://docs.ray.io/en/latest/tune/examples/pbt_guide.html
# api docs -> https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html#ray.tune.schedulers.PopulationBasedTraining
# ========================

# perturbution_interval the frequency at which hyperparameters are perturbed or modified within the population.
# Setting it too frequently can lead to excessive overhead from checkpointing models for evaluation and mutation.
# Setting it too infrequently might slow down the process of finding optimal hyperparameters.
perturbation_interval = 4  # see notes above 
scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=perturbation_interval,
    metric="eval_accuracy",
    mode="max",
    hyperparam_mutations={
        "train_loop_config": {
            "weight_decay": tune.uniform(0.0, 0.03),
            "learning_rate": tune.loguniform(5e-5, 2e-3),
            # Tests have shown weight_decay and learning_rate are two of the most import params to tune. 
            # you can also try tuning other hyperparams, such as batch size, grad accumulation steps and optim, and scheduler. 
        }
    },
)

# Set up the Tuner
# api docs -> https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html#ray.tune.Tuner
# ========================
tuner = Tuner(
    trainer,
    param_space={  # this is search space for the tuning job. `train_loop_config` passed as `config` to `TorchTrainer`
        "train_loop_config": {
            "learning_rate": tune.loguniform(5e-5, 2e-3),
            "weight_decay": tune.uniform(0.0, 0.03),
        }
    },
    # see: https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray.train.RunConfig for details
    run_config=train.RunConfig(
        name="pbt_mamba_finacial_phrasebank_sentiment",
        # Stop when we've reached a threshold accuracy, or a maximum
        # training_iteration, whichever comes first
        # a training interation is done after the number of samples trains
        stop={"eval_accuracy": 0.90, "training_iteration": 20}, 
        checkpoint_config=CheckpointConfig(
            checkpoint_score_attribute="eval_accuracy",
            num_to_keep=4,
            checkpoint_score_order="max",
        ),
        storage_path="/files/tmp/ray_results",
    ),
    # see https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html for details
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=4, 
        reuse_actors=True,
    )
)

# Fit the model
result = tuner.fit()

# Finally we load the best checkpoint and saving it in safe tensors.
# ========================
best_result = result.get_best_result("eval_accuracy", mode="max")
print('Best Result:\n==================')
print(f'{best_result}')

# with best_result.checkpoint.as_directory() as checkpoint_dir:
#     print('checkpoint_dir:', checkpoint_dir)
    # model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
# trainer.save_model(output_dir=f'/files/training/mamba_finacial_phrasebank_sentiment/complete')