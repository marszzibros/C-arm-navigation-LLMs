import os


os.environ["UNSLOTH_DISABLE_PATCHING"] = "1"  
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # disables compile wrappers, keeps other speedups
os.environ["TORCH_COMPILE_DISABLE"] = "1"    # or use TORCH_COMPILE_DEBUG=1 if you want tracing info
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


from unsloth import FastVisionModel 
import pandas as pd
from transformers import TextIteratorStreamer
import threading
from PIL import Image
import os
import time



class Inference:
    def __init__(self, model_id, mode="test"):   
        self.model_id = model_id
        
        
        if mode == "test":
            self.folder = "stage1_test_results"
            os.system(f"mkdir {self.folder}")
            self.df = pd.read_csv("data/test.csv",index_col=0)
            # per patient, sample 256 images
            self.df = self.df.groupby('case_number').sample(n=256, random_state=42).reset_index(drop=True)
            with open("prompts/top_3_closest_prompt.txt", "r") as f:
                self.system_prompt = f.read()
            self.user_prompt = "Please classify the three closest landmarks based on the text, returning them in the exact format [landmark_index1: landmark_name1, landmark_index2: landmark_name2, landmark_index3: landmark_name3] and do not include any extra text."
            
        
        elif mode == "classification":
            self.folder = "stage1_classification_results"
            os.system(f"mkdir {self.folder}")
            self.df = pd.read_csv("data/classification.csv",index_col=0)
            with open("prompts/classification_prompt.txt", "r") as f:
                self.system_prompt = f.read()
            self.user_prompt = "I just took this C-arm shot. Can you tell me which landmarks are near the center? Two would be enough."
            
        
        self.model, self.processor = FastVisionModel.from_pretrained(
            model_name=self.model_id,
            load_in_4bit=False,
        )
        FastVisionModel.for_inference(self.model)
        print("Model and processor loaded.")


        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.user_prompt}
            ]}
        ]

    def inference(self):
        csv_path = f"{self.model_id}.csv"
        csv_path = csv_path.replace("/", "_")
        csv_path = os.path.join(self.folder, csv_path)
        file_exists = os.path.exists(csv_path)

        for i, row in self.df.iterrows():
            image_path = row['filename']
            image = Image.open(image_path).convert("RGB").resize((512, 512))

            input_text = self.processor.apply_chat_template(
                self.messages,
                add_generation_prompt=True,
            )
            # print(input_text )
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")


            # --- measure start time ---
            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=1, top_p=0.95, top_k=64,
            )
            generated_text = self.processor.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            # # --- run inference with streaming ---
            # streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True)
            # thread = threading.Thread(
            #     target=self.model.generate,
            #     kwargs=dict(**inputs, streamer=streamer, max_new_tokens=128,
            #                 use_cache=True, temperature = 1.0, top_k = 64, top_p = 0.95, min_p = 0.0)
            # )
            
            # thread.start()

            # --- record end time & compute duration ---
            end_time = time.time()
            elapsed = round(end_time - start_time, 2)  # seconds (2 decimal places)

            # --- create one-row DataFrame ---
            df_row = pd.DataFrame([{
                "filename": image_path,
                "output": generated_text,
                "inference_time_sec": elapsed
            }])

            # --- append to CSV ---
            df_row.to_csv(
                csv_path,
                mode='a',
                header=not file_exists,
                index=False,
                encoding="utf-8"
            )
            file_exists = True

            print(f"✅ Saved output for {image_path} ({elapsed}s)")