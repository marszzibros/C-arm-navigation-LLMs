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
from trl import SFTConfig, SFTTrainer

from omegaconf import  OmegaConf
torch._inductor.config.compile_threads = 1
torch._dynamo.config.suppress_errors = True

class Utils:
    def __init__(self):
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

class Model:
    def __init__(self, cfg):
        self.utils = Utils()
        self.cfg = cfg
        self.model, self.processor = FastVisionModel.from_pretrained(
            cfg.model_id,
            load_in_4bit = True,
            max_seq_length =16384,
            use_gradient_checkpointing = "unsloth",
            gpu_memory_utilization = 0.8,
            device_map = "balanced",
            
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
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers     = True, # False if not finetuning vision layers
                finetune_language_layers   = True, # False if not finetuning language layers
                finetune_attention_modules = True, # False if not finetuning attention layers
                finetune_mlp_modules       = False, # False if not finetuning MLP layers

                r = lora_config_dict['r'],           # The larger, the higher the accuracy, but might overfit
                lora_alpha = lora_config_dict['r'] * 2,  # Recommended alpha == r at least
                lora_dropout = lora_config_dict['lora_dropout'],
                bias = lora_config_dict['bias'],
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
                use_gradient_checkpointing = "unsloth", # Reduces memory usage
            )
    def SFT(self):

        training_args = SFTConfig(**self.train_config_dict)
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            data_collator = UnslothVisionDataCollator(self.model, self.processor),
            train_dataset=self.dataset,

        )
        self.trainer.train()
        self.model.save_pretrained_merged(self.cfg.train.output_dir+str(16), self.processor, save_method = "merged_16bit",)
