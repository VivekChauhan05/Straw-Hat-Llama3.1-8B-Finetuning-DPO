# DPO 
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import PatchDPOTrainer, FastLanguageModel, is_bfloat16_supported
PatchDPOTrainer()




def dpo_train():   
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "hugging_face_user_id_name/model_name", # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    #@title Alignment Handbook utils
    import os
    import re
    from typing import List, Literal, Optional
    
    from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
    from datasets.builder import DatasetGenerationError
    
    
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    
    
    def apply_chat_template(
        example, tokenizer, task: Literal["sft", "generation", "rm", "dpo"] = "sft", assistant_prefix="<|assistant|>\n"
    ):
        def _strip_prefix(s, pattern):
            # Use re.escape to escape any special characters in the pattern
            return re.sub(f"^{re.escape(pattern)}", "", s)
    
        if task in ["sft", "generation"]:
            messages = example["messages"]
            # We add an empty system message if there is none
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
            example["text"] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
            )
        elif task == "rm":
            if all(k in example.keys() for k in ("chosen", "rejected")):
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
                # We add an empty system message if there is none
                if chosen_messages[0]["role"] != "system":
                    chosen_messages.insert(0, {"role": "system", "content": ""})
                if rejected_messages[0]["role"] != "system":
                    rejected_messages.insert(0, {"role": "system", "content": ""})
                example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
                example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            else:
                raise ValueError(
                    f"Could not format example as dialogue for rm task! Require [chosen, rejected] keys but found {list(example.keys())}"
                )
        elif task == "dpo":
            if all(k in example.keys() for k in ("chosen", "rejected")):
                # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
                prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
                # Insert system message
                if example["chosen"][0]["role"] != "system":
                    prompt_messages.insert(0, {"role": "system", "content": ""})
                else:
                    prompt_messages.insert(0, example["chosen"][0])
                # TODO: handle case where chosen/rejected also have system messages
                chosen_messages = example["chosen"][1:]
                rejected_messages = example["rejected"][1:]
                example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
                example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
                example["text_prompt"] = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
                example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
            else:
                raise ValueError(
                    f"Could not format example as dialogue for dpo task! Require [chosen, rejected] keys but found {list(example.keys())}"
                )
        else:
            raise ValueError(
                f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
            )
        return example
    
    
    def get_datasets(
        data_config: dict,
        splits: List[str] = ["train", "test"],
        shuffle: bool = True,
    ) -> DatasetDict:
        """
        Loads one or more datasets with varying training set proportions.
    
        Args:
            data_config (DataArguments or dict):
                Dataset configuration and split proportions.
            splits (List[str], *optional*, defaults to ['train', 'test']):
                Dataset splits to load and mix. Assumes the splits exist in all datasets and have a train_ or test_ prefix.
            shuffle (bool, *optional*, defaults to True):
                Whether to shuffle the training and testing/validation data.
    
        Returns
            [DatasetDict]: The dataset dictionary containing the loaded datasets.
        """
    
        if type(data_config) is dict:
            # Structure of the input is:
            #     dataset_mixer = {
            #             "dataset1": 0.5,
            #             "dataset1": 0.3,
            #             "dataset1": 0.2,
            #         }
            dataset_mixer = data_config
        else:
            raise ValueError(f"Data config {data_config} not recognized.")
    
        raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
        return raw_datasets
    
    
    def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
        """
        Loads and mixes datasets according to proportions specified in dataset_mixer.
    
        Args:
            dataset_mixer (dict):
                Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
            splits (Optional[List[str]], *optional*, defaults to None):
                Dataset splits to load and mix. Assumes the splits exist in all datasets and have a train_ or test_ prefix.
            shuffle (bool, *optional*, defaults to True):
                Whether to shuffle the training and testing/validation data.
        """
        raw_datasets = DatasetDict()
        raw_train_datasets = []
        raw_val_datasets = []
        fracs = []
        for ds, frac in dataset_mixer.items():
            fracs.append(frac)
            for split in splits:
                try:
                    # Try first if dataset on a Hub repo
                    dataset = load_dataset(ds, split=split)
                except DatasetGenerationError:
                    # If not, check local dataset
                    dataset = load_from_disk(os.path.join(ds, split))
    
                if "train" in split:
                    raw_train_datasets.append(dataset)
                elif "test" in split:
                    raw_val_datasets.append(dataset)
                else:
                    raise ValueError(f"Split type {split} not recognized as one of test or train.")
    
        if any(frac < 0 for frac in fracs):
            raise ValueError("Dataset fractions cannot be negative.")
    
        if len(raw_train_datasets) > 0:
            train_subsets = []
            for dataset, frac in zip(raw_train_datasets, fracs):
                train_subset = dataset.select(range(int(frac * len(dataset))))
                train_subsets.append(train_subset)
            if shuffle:
                raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
            else:
                raw_datasets["train"] = concatenate_datasets(train_subsets)
        # No subsampling for test datasets to enable fair comparison across models
        if len(raw_val_datasets) > 0:
            if shuffle:
                raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
            else:
                raw_datasets["test"] = concatenate_datasets(raw_val_datasets)
    
        if len(raw_datasets) == 0:
            raise ValueError(
                f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
            )
    
        return raw_datasets
    
    
    raw_datasets = get_datasets(
        {"HuggingFaceH4/ultrafeedback_binarized" : 0.5}, # 50% sampled
        splits = ["train_prefs", "test_prefs"],
    )
    column_names = list(raw_datasets["train"].features)
    
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs = {"tokenizer": tokenizer, "task": "dpo"},
        num_proc = 12,
        remove_columns = column_names,
        desc = "Formatting comparisons with prompt template",
    )
    
    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
    
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Currently only supports dropout = 0
        bias = "none",    # Currently only supports bias = "none"
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # rank stabilized LoRA
        loftq_config = None, # LoftQ
    )
    
    
    
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 2,
            gradient_checkpointing = True,
            warmup_ratio = 0.1,
            num_train_epochs = 1,
            max_steps = 200,
            learning_rate = 2e-6,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.0,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs",
        ),
        beta = 0.1,
        train_dataset = raw_datasets["train"],
        # eval_dataset = raw_datasets["test"],
        tokenizer = tokenizer,
        max_length = 1024,
        max_prompt_length = 512,
    )
    
    dpo_trainer.train()
    
    # %load_ext tensorboard
    # %tensorboard --logdir outputs/runs
    
    model.save_pretrained_merged("hugging_face_user_id_name/model_name", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged("hugging_face_user_id_name/model_name", tokenizer, save_method="merged_16bit", token = "your_huggingface_token")
    
if __name__ == "__main__":
    dpo_train()
    
    