import torch
from config import *
from tokenizer.tokenizer import Vocabulary
from model.transformer import AURATransformer
from validator.validator import build_validator
def load():
 v=Vocabulary.load(TOKENIZER_FILE); c=torch.load(CHECKPOINT_FILE,map_location=DEVICE,weights_only=False); m=AURATransformer(len(v),D_MODEL,N_HEADS,NUM_ENCODER_LAYERS,NUM_DECODER_LAYERS,DIM_FEEDFORWARD,DROPOUT,MAX_LEN,v.pad_id).to(DEVICE); m.load_state_dict(c['model_state_dict']); m.eval(); return m,v
@torch.no_grad()
def translate(m,v,text):
 src=torch.tensor([v.encode(text)[:MAX_LEN]],device=DEVICE); out=[v.bos_id]
 for _ in range(MAX_LEN-1):
  logits=m(src,torch.tensor([out],device=DEVICE)); n=logits[:,-1].argmax().item(); out.append(n)
  if n==v.eos_id: break
 return v.decode(out)
def main():
 m,v=load(); n=build_validator(TRAIN_FILE); print('AURA v0.1 — English → Oshindonga'); print("Type 'quit' to exit.")
 while True:
  s=input('\nEnglish > ').strip()
  if s.lower()=='quit': break
  t=translate(m,v,s); print('Oshindonga >',t); print(f'N-gram score: {n.score(t):.4f}')
if __name__=='__main__': main()
