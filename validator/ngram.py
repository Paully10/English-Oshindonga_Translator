import math
from collections import Counter
class CharacterNGram:
 def __init__(self,n=4,alpha=.1): self.n=n; self.alpha=alpha; self.counts=Counter(); self.context=Counter(); self.alphabet=set()
 def fit(self,texts):
  for text in texts:
   s=f'^{text.lower().strip()}$'
   for i in range(len(s)-self.n+1):
    g=s[i:i+self.n]; self.counts[g]+=1; self.context[g[:-1]]+=1; self.alphabet.add(g[-1])
  return self
 def score(self,text):
  s=f'^{text.lower().strip()}$'; grams=[s[i:i+self.n] for i in range(len(s)-self.n+1)]; V=max(1,len(self.alphabet)); return sum(math.log((self.counts[g]+self.alpha)/(self.context[g[:-1]]+self.alpha*V)) for g in grams)/max(1,len(grams))
