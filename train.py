from huggingface_hub import login

import hydra
from omegaconf import DictConfig, OmegaConf
from Trainer import Model

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    cfg.train.output_dir = cfg.train.output_dir + "_"+ cfg.model_id.split("/")[1].split("-")[0]+ "_" + cfg.model_id.split("/")[1].split("-")[-2] + "_" + cfg.mode
    print(OmegaConf.to_yaml(cfg))

    trainer = Model(cfg)
    trainer.SFT()

        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback, sys
        print("Error occurred:", e)
        traceback.print_exc()
        sys.exit(1)
