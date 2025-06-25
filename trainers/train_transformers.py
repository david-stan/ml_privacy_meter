from typing import Tuple, Dict
import os
import matplotlib.pyplot as plt
import pandas as pd

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

_tokenizer_cache: dict[str, AutoTokenizer] = {}


def create_training_args(configs: Dict) -> TrainingArguments:
    """Creates and returns the training arguments for the transformer model."""
    return TrainingArguments(
        output_dir=configs["run"]["log_dir"],
        num_train_epochs=configs["train"]["epochs"],
        per_device_train_batch_size=configs["train"]["batch_size"],
        per_device_eval_batch_size=configs["train"]["batch_size"],
        auto_find_batch_size=configs["train"].get(
            "auto_find_batch_size", False
        ),  # if this is not specified in the config, use the specified batch size
        warmup_steps=500,
        optim=configs["train"]["optimizer"],
        weight_decay=configs["train"]["weight_decay"],
        learning_rate=configs["train"]["learning_rate"],
        save_strategy="no",
        logging_strategy="steps",
        eval_strategy="steps",
        logging_steps=50,
        eval_steps=50,
        save_total_limit=1,
        gradient_accumulation_steps=configs["train"].get(
            "gradient_accumulation_steps", 1
        ),
        fp16=True,
    )

def setup_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """Loads the tokenizer and ensures pad token is set."""
    if tokenizer_name in _tokenizer_cache:
        return _tokenizer_cache[tokenizer_name]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, clean_up_tokenization_spaces=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(
            "The tokenizer pad token is None. Setting it to the EOS token for padding. "
            "If this is not desired, please set the pad token manually."
        )
    _tokenizer_cache[tokenizer_name] = tokenizer
    return tokenizer


def logging(trainer, training_args, model_idx):
    log_history = trainer.state.log_history
    train_logs = [log for log in log_history if 'loss' in log and 'step' in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log and 'step' in log]

    train_steps = [log['step'] for log in train_logs]
    train_losses = [log['loss'] for log in train_logs]
    eval_steps = [log['step'] for log in eval_logs]
    eval_losses = [log['eval_loss'] for log in eval_logs]

    output_dir = training_args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create and save plot
    plt.figure()
    plt.plot(train_steps, train_losses, label="train_loss")
    plt.plot(eval_steps, eval_losses, label="eval_loss")
    plt.title("Training and Evaluation Loss History")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"loss_history_{model_idx}.png"))
    plt.close()

    # Save loss history to CSV for further analysis
    train_df = pd.DataFrame({'step': train_steps, 'train_loss': train_losses})
    eval_df = pd.DataFrame({'step': eval_steps, 'eval_loss': eval_losses})
    loss_df = pd.merge(train_df, eval_df, on='step', how='outer').sort_values('step')
    loss_df.to_csv(os.path.join(output_dir, f"loss_history_{model_idx}.csv"), index=False)


def train_transformer(
    trainset, model: PreTrainedModel, configs: Dict, testset, model_idx
) -> Tuple[PreTrainedModel, float, float]:
    """Train a Hugging Face transformer model without any PEFT (LoRA) modifications."""
    if not isinstance(model, PreTrainedModel):
        raise ValueError("The provided model is not a Hugging Face transformer model")

    training_args = create_training_args(configs)
    tokenizer = setup_tokenizer(configs["data"]["tokenizer"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
    )

    trainer.train()

    logging(trainer, training_args, model_idx)

    train_loss = trainer.state.log_history[-1]["train_loss"]
    test_loss = trainer.state.log_history[-2]["eval_loss"]

    return model, train_loss, test_loss


def get_peft_model_config(configs: Dict) -> LoraConfig:
    """Get the PEFT model configuration."""
    if "peft" not in configs["train"]:
        raise ValueError("LoRA configuration is not provided in the configuration file")

    if configs["train"]["peft"]["type"] == "lora":
        # Change the lora field in the config file to change these parameters
        return LoraConfig(
            fan_in_fan_out=configs["train"]["peft"]["fan_in_fan_out"],
            inference_mode=False,
            r=configs["train"]["peft"]["r"],
            target_modules=configs["train"]["peft"]["target_modules"],
            lora_alpha=32,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        raise NotImplementedError("Only LoRA is supported in this implementation.")


def train_transformer_with_peft(
    trainset, peft_model, configs: Dict, testset
) -> Tuple[PreTrainedModel, float, float]:
    """Train a Hugging Face transformer model with PEFT (LoRA) modifications."""
    # if not isinstance(model, PreTrainedModel):
    #     raise ValueError("The provided model is not a Hugging Face transformer model")
    #
    # # Apply PEFT (LoRA) configuration
    # peft_config = get_peft_model_config(configs)
    # peft_model = get_peft_model(model, peft_config)

    training_args = create_training_args(configs)
    tokenizer = setup_tokenizer(configs["data"]["tokenizer"])

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
    )

    trainer.train()
    train_loss = trainer.state.log_history[-1]["train_loss"]
    test_loss = trainer.state.log_history[-2]["eval_loss"]

    return peft_model, train_loss, test_loss
