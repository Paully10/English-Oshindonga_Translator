import json,torch
from torch.utils.data import Dataset,DataLoader
from tokenizer.tokenizer import Vocabulary
from config import MAX_LEN
class TranslationDataset(Dataset):
 def __init__(self,path,vocab): self.rows=[(json.loads(x)['source'],json.loads(x)['target']) for x in open(path,encoding='utf8')]; self.vocab=vocab
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  s,t=self.rows[i]; return torch.tensor(self.vocab.encode(s)[:MAX_LEN]),torch.tensor(self.vocab.encode(t)[:MAX_LEN])
def collate(pad_id):
 def f(batch):
  ss,tt=zip(*batch); sm=max(map(len,ss)); tm=max(map(len,tt)); S=torch.full((len(batch),sm),pad_id,dtype=torch.long); T=torch.full((len(batch),tm),pad_id,dtype=torch.long)
  for i,x in enumerate(ss): S[i,:len(x)]=x
  for i,x in enumerate(tt): T[i,:len(x)]=x
  return S,T
 return f
def make_loader(path,vocab,batch_size,shuffle): return DataLoader(TranslationDataset(path,vocab),batch_size=batch_size,shuffle=shuffle,collate_fn=collate(vocab.pad_id))
