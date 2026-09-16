import random,numpy as np,torch,torch.nn as nn
from tqdm import tqdm
from config import *
from tokenizer.tokenizer import Vocabulary
from training.dataset import make_loader
from model.transformer import AURATransformer
def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
def epoch(model,loader,loss_fn,opt=None):
 train=opt is not None; model.train(train); total=0
 with (torch.enable_grad() if train else torch.no_grad()):
  for src,tgt in tqdm(loader,leave=False):
   src,tgt=src.to(DEVICE),tgt.to(DEVICE); logits=model(src,tgt[:,:-1]); loss=loss_fn(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1))
   if train: opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_GRAD_NORM); opt.step()
   total+=loss.item()
 return total/max(1,len(loader))
def main():
 seed(SEED); v=Vocabulary.load(TOKENIZER_FILE); tr=make_loader(TRAIN_FILE,v,BATCH_SIZE,True); va=make_loader(VAL_FILE,v,BATCH_SIZE,False); m=AURATransformer(len(v),D_MODEL,N_HEADS,NUM_ENCODER_LAYERS,NUM_DECODER_LAYERS,DIM_FEEDFORWARD,DROPOUT,MAX_LEN,v.pad_id).to(DEVICE); loss=nn.CrossEntropyLoss(ignore_index=v.pad_id); opt=torch.optim.AdamW(m.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY); best=float('inf'); bad=0; print('Device:',DEVICE); print('Vocabulary:',len(v)); print('Parameters:',sum(p.numel() for p in m.parameters()))
 for e in range(1,EPOCHS+1):
  a=epoch(m,tr,loss,opt); b=epoch(m,va,loss); print(f'Epoch {e:02d}/{EPOCHS} | train={a:.4f} | val={b:.4f}')
  if b<best: best=b; bad=0; torch.save({'model_state_dict':m.state_dict()},CHECKPOINT_FILE); print('  Best checkpoint saved.')
  else: bad+=1; 
  if bad>=PATIENCE: print('Early stopping.'); break
if __name__=='__main__': main()
