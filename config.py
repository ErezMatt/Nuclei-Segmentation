import yaml
import torch

with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE = config["learning_rate"]
BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
NUM_WORKERS = config["num_workers"]
PIN_MEMORY = config["pin_memory"]
PATCH_SIZE = config["patch_size"]
IMG_SIZE = config["img_size"]

TRAIN_DIR_PATH = config["train_dir"]
TEST_DIR_PATH = config["test_dir"]

VIT_PATCH_SIZE = config["vit_patch_size"]
VIT_EMBED_DIM = config["vit_embed_dim"]
VIT_DEPTH = config["vit_depth"]
VIT_NUM_HEADS = config["vit_num_heads"]

CHECKPOINT_DIR = "checkpoints"