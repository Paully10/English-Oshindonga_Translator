from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/'data'; CHECKPOINT_DIR=ROOT/'checkpoints'; CHECKPOINT_DIR.mkdir(exist_ok=True)
TRAIN_FILE=DATA_DIR/'train.jsonl'; VAL_FILE=DATA_DIR/'validation.jsonl'; TEST_FILE=DATA_DIR/'test.jsonl'
TOKENIZER_FILE=CHECKPOINT_DIR/'tokenizer.json'; CHECKPOINT_FILE=CHECKPOINT_DIR/'aura_oshindonga_v0_1.pt'
SEED=42; MIN_FREQ=1; MAX_VOCAB_SIZE=12000
D_MODEL=256; N_HEADS=8; NUM_ENCODER_LAYERS=4; NUM_DECODER_LAYERS=4; DIM_FEEDFORWARD=1024; DROPOUT=.1; MAX_LEN=128
BATCH_SIZE=32; EPOCHS=30; LEARNING_RATE=3e-4; WEIGHT_DECAY=1e-4; CLIP_GRAD_NORM=1.; PATIENCE=5
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
