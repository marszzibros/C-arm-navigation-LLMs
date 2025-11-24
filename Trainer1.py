import os
import re
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import random

import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset
import torch

from unsloth import FastVisionModel, UnslothVisionDataCollator, get_chat_template
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer

from omegaconf import  OmegaConf
torch._inductor.config.compile_threads = 1
torch._dynamo.config.suppress_errors = True

class Utils:
    def __init__(self, processor):
        self.processor = processor
        self.landmarks = {
            1: ["Skull", "Cranium", "Cranial vault", "Calvarium"],
            2: ["Right humeral head", "Right Proximal humerus", "Head of right humerus", "Right glenohumeral head"],
            3: ["Left humeral head", "Left Proximal humerus", "Head of left humerus", "Left glenohumeral head"],
            4: ["Right scapular inferior angle", "Lower tip of right scapula", "Right shoulder blade tip"],
            5: ["Left scapular inferior angle", "Lower tip of left scapula", "Left shoulder blade tip"],
            6: ["Right elbow", "Right olecranon", "Right Cubital joint", "Right Elbow joint"],
            7: ["Left elbow", "Left olecranon", "Left Cubital joint", "Left Elbow joint"],
            8: ["Right wrist", "Right carpus", "Right Radiocarpal joint", "Right Wrist joint"],
            9: ["Left wrist", "Left carpus", "Left Radiocarpal joint", "Left Wrist joint"],
            10: ["T1", "First thoracic vertebra", "T1 vertebral body", "Upper thoracic vertebra"],
            11: ["Carina", "Tracheal bifurcation", "Tracheal carina"],
            12: ["Right hemidiaphragm", "Right Diaphragmatic dome", "Right Diaphragmatic crus"],
            13: ["Left hemidiaphragm", "Left Diaphragmatic dome", "Left Diaphragmatic crus"],
            14: ["T12", "Twelfth thoracic vertebra", "T12 vertebral body", "Lower thoracic vertebra"]
        }
    def load_image(self, examples):
        images = []
        for path in examples["file_name"]:
            img = Image.open(path).convert("RGB").resize((512, 512))
            images.append(img)
        examples["image"] = images
        return examples
    def format_rl_data(self, examples):
        with open("prompts/top_3_closest_prompt.txt", "r") as f:
            system_prompt = f.read()

        user_prompt = "Please classify the three closest landmarks based on the text, returning them in the exact format [landmark_index1: landmark_name1, landmark_index2: landmark_name2, landmark_index3: landmark_name3] and do not include any extra text."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        examples["prompt"] = [messages for _ in range(len(examples["file_name"]))]
        return examples
    def grpo_dataset(self):
        df = pd.read_csv("data/grpo.csv", index_col=0)
        df['ground_truth'] = df['top3_landmarks'].apply(lambda x: np.array(x[1:-1].split(", "),dtype=int))
        dataset = Dataset.from_pandas(df)

        dataset = dataset.map(self.load_image, batched=True, num_proc=32)
        dataset = dataset.map(self.format_rl_data, batched=True, num_proc=32)
        return dataset
    def format_data(self, sample):
        
        with open("prompts/top_3_closest_prompt.txt", "r") as f:
            system_prompt = f.read()

        user_prompt = "Please classify the three closest landmarks based on the text, returning them in the exact format [landmark_index1: landmark_name1, landmark_index2: landmark_name2, landmark_index3: landmark_name3] and do not include any extra text."
        answer = f"[{sample['answer'][0]}: {random.choice(self.landmarks[sample['answer'][0]])}, {sample['answer'][1]}: {random.choice(self.landmarks[sample['answer'][1]])}, {sample['answer'][2]}: {random.choice(self.landmarks[sample['answer'][2]])}]"

        messages = [
            {"role": "system", "content": [{"type":"text", "text":system_prompt}]},
            {"role": "user", "content": [{"type":"text", "text":user_prompt}, {"type":"image","image":sample["image"]}]},
            {"role": "assistant", "content": [{"type":"text", "text":answer}]}
        ]

        return {"messages" : messages}


    def sft_datasets(self, folder_path="data"):
        # Load your CSV into a pandas DataFrame
        df = pd.read_csv(os.path.join(folder_path, "sft.csv"), index_col=0)

        # Convert pandas DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)

        # Define your batched image processing function
        def process_batch(examples):
            images = []
            answers = []
            for path, ans_str in zip(examples["file_name"], examples["top3_landmarks"]):
                # Load and preprocess image
                img = Image.open(path).convert("RGB").resize((896, 896))
                images.append(img)

                # Convert answer string to numpy array
                ans = np.array(ans_str[1:-1].split(", "), dtype=int)
                answers.append(ans)

            return {"image": images, "answer": answers}

        # Use map for parallelized processing
        dataset = dataset.map(
            process_batch,
            batched=True,
            num_proc=32,  # adjust depending on CPU availability
            desc="Processing images and labels"
        )
        datasets = []
        for i in range(len(dataset)):
            formatted = self.format_data(dataset[i])
            datasets.append(formatted)  
        print("done formatting data")
        

        return datasets
        
    def reward(self, completions, ground_truth, **kwargs):

        rewards = []

        for completion, truth in zip(completions, ground_truth):
            try:

                completion_text = completion[0]["content"] if isinstance(completion, list) else completion
                ground_truth = truth
                completion_text = completion_text.strip()

                # 1. Validate format
                if not (completion_text.startswith("[") and completion_text.endswith("]")):
                    rewards.append(-10.0)
                    continue
                # 2. Parse items
                samples = re.split(r',\s*', completion_text[1:-1].strip())
                truths = ground_truth

                if len(samples) != 3:
                    rewards.append(-5.0)
                    continue

                reward = 1.0  # base reward for correct format
                
                # 3. Extract key/value pairs
                def split_item(x): return x.split(": ", 1)
                keys, values = zip(*[split_item(s) for s in samples])
                # 4. Matching keys
                reward += len(set(keys) & set(list(truths)))

                # 5. Key-value correctness
                for k, v, tk in zip(keys, values,truths):

                    if int(k) == tk and int(k) in self.landmarks:
                        landmark_val = self.landmarks[int(k)]
                        if isinstance(landmark_val, (list, set, tuple)):
                            if v.lower() in [item.lower() for item in landmark_val]:
                                reward += 1
                        elif isinstance(landmark_val, str):
                            if v.lower() == landmark_val.lower():
                                reward += 1

                rewards.append(float(reward))

            except Exception as e:
                # Log once per sample
                rewards.append(-10.0)
        
        return rewards
class Model:
    def __init__(self, cfg):
        self.utils = Utils(None)
        self.cfg = cfg
        self.model, self.processor = FastVisionModel.from_pretrained(
            cfg.model_id,
            #load_in_4bit = True,
            max_seq_length =16384,
            use_gradient_checkpointing = "unsloth",
            gpu_memory_utilization = 0.8,
            # device_map = "balanced",
            full_finetuning = True
            
        )
        lora_config_dict = OmegaConf.to_container(cfg.lora, resolve=True)
        self.train_config_dict = OmegaConf.to_container(cfg.train, resolve=True)
        # dataset
        if "SFT" in cfg.train.output_dir:
            self.dataset = self.utils.sft_datasets()
            if "gemma" in cfg.train.output_dir:
                
                self.processor = get_chat_template(
                    self.processor,
                    "gemma-3"
                )
            self.utils.processor = self.processor

        elif "GRPO" in cfg.train.output_dir:
            if "gemma" in cfg.train.output_dir:
                self.train_config_dict['max_prompt_length'] = 512
            self.dataset = self.utils.grpo_dataset()


        # if "all_linear" in cfg.train.output_dir:
        #     self.model = FastVisionModel.get_peft_model(
        #         self.model,
        #         finetune_vision_layers     = True, # False if not finetuning vision layers
        #         finetune_language_layers   = True, # False if not finetuning language layers
        #         finetune_attention_modules = True, # False if not finetuning attention layers
        #         finetune_mlp_modules       = True, # False if not finetuning MLP layers

        #         r = lora_config_dict['r'],           # The larger, the higher the accuracy, but might overfit
        #         lora_alpha = lora_config_dict['r'] * 2,  # Recommended alpha == r at least
        #         lora_dropout = lora_config_dict['lora_dropout'],
        #         bias = lora_config_dict['bias'],
        #         random_state = 3407,
        #         use_rslora = False,  # We support rank stabilized LoRA
        #         loftq_config = None, # And LoftQ
        #         use_gradient_checkpointing = "unsloth", # Reduces memory usage
        #     )
        # else:
        #     self.model = FastVisionModel.get_peft_model(
        #         self.model,
        #         finetune_vision_layers     = True, # False if not finetuning vision layers
        #         finetune_language_layers   = True, # False if not finetuning language layers
        #         finetune_attention_modules = True, # False if not finetuning attention layers
        #         finetune_mlp_modules       = False, # False if not finetuning MLP layers

        #         r = lora_config_dict['r'],           # The larger, the higher the accuracy, but might overfit
        #         lora_alpha = lora_config_dict['r'] * 2,  # Recommended alpha == r at least
        #         lora_dropout = lora_config_dict['lora_dropout'],
        #         bias = lora_config_dict['bias'],
        #         random_state = 3407,
        #         use_rslora = False,  # We support rank stabilized LoRA
        #         loftq_config = None, # And LoftQ
        #         use_gradient_checkpointing = "unsloth", # Reduces memory usage
        #     )

    def GRPO(self):
        training_args = GRPOConfig(**self.train_config_dict, importance_sampling_level="sequence")
        self.trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            reward_funcs=self.utils.reward,
            train_dataset=self.dataset,
        )
        self.trainer.train()
        
        self.model.save_pretrained_merged(self.cfg.train.output_dir+str(16), self.processor, save_method = "merged_16bit",)
        self.trainer.save_model()
    def SFT(self):

        training_args = SFTConfig(**self.train_config_dict)
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            data_collator = UnslothVisionDataCollator(self.model, self.processor),
            train_dataset=self.dataset,

        )
        self.trainer.train()
        self.model.save_pretrained(self.cfg.train.output_dir)
        self.processor.save_pretrained(self.cfg.train.output_dir + str("_processor"))
        
        
        # self.model.save_pretrained_merged(self.cfg.train.output_dir+str(8), self.processor, save_method = "merged_8bit",)
        # self.trainer.save_model()