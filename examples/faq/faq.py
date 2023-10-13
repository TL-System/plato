from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from peft import PeftModel
import os
import logging

os.environ["WANDB_DISABLED"] = "true"


def load_data(datasource):
    train_dataset = load_dataset(datasource, split="train[0:50]")
    eval_dataset = load_dataset(datasource, split="train[50:]")

    return train_dataset, eval_dataset


def formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


def load_base_model(base_model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model


def load_tokenizer(base_model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_data(train_dataset, eval_dataset, tokenizer):
    def generate_and_tokenize_prompt(prompt):
        result = tokenizer(
            formatting_func(prompt),
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        # result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_eval_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    return tokenized_train_dataset, tokenized_eval_dataset


def inference(prompt, model, tokenizer):
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        print(
            tokenizer.decode(
                model.generate(**model_input, max_new_tokens=128, pad_token_id=2)[0],
                skip_special_tokens=True,
            )
        )


def setup_adapter(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # Apply the accelerator
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=False
        ),
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare_model(model)

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def finetune(
    model,
    tokenizer,
    tokenized_train_dataset,
    tokenized_eval_dataset,
    output_dir,
    logging_dir,
):
    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=500,
            learning_rate=2.5e-5,  # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir=logging_dir,  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=50,  # Save checkpoints every 50 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=50,  # Evaluate and save checkpoints every 50 steps
            do_eval=True,  # Perform evaluation at the end of training
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()


def load_ft_model(base_model_id, ckpt_name):
    base_model = load_base_model(base_model_id)
    ft_model = PeftModel.from_pretrained(base_model, ckpt_name)
    return ft_model


def main():
    train = True
    datasource = "CShorten/CDC-COVID-FAQ"
    base_model_id = "meta-llama/Llama-2-7b-hf"
    # 1. Load data
    train_dataset, eval_dataset = load_data(datasource)
    # 2. Load model
    base_model = load_base_model(base_model_id)
    # 3. Tokenization
    tokenizer = load_tokenizer(base_model_id)
    tokenized_train_dataset, tokenized_eval_dataset = tokenize_data(
        train_dataset, eval_dataset, tokenizer
    )
    # 4. Evaluate base model
    eval_prompt = "### Question: How are COVID-19 patients treated?\n ### Answer: "
    logging.info("Evaluate base model with an example prompt...")
    inference(eval_prompt, base_model, tokenizer)
    # 5. Set up LoRA and run training
    base_dir = os.path.abspath(os.path.dirname(__file__))
    base_model_name = "llama2-7b"
    output_dir = base_dir + "/" + base_model_name + "-finetune"
    logging_dir = base_dir + "/logs"
    if train:
        model = setup_adapter(base_model)
        finetune(
            model,
            tokenizer,
            tokenized_train_dataset,
            tokenized_eval_dataset,
            output_dir,
            logging_dir,
        )
    # 6. Evaluate fine-tuned model
    num_steps = 50
    ckpt_name = output_dir + "/checkpoint-" + str(num_steps)
    ft_model = load_ft_model(base_model_id, ckpt_name)
    logging.info("Evaluate fine-tuned model with the same example prompt...")
    inference(eval_prompt, ft_model, tokenizer)


if __name__ == "__main__":
    main()
