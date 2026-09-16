import json
from validator.ngram import CharacterNGram
def build_validator(path):
 m=CharacterNGram(); m.fit(json.loads(x)['target'] for x in open(path,encoding='utf8')); return m
