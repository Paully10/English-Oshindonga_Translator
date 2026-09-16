"""
Consolidates every Oshindonga data source into one canonical dictionary.

Sources merged (Oshindonga only — Kwanyama file excluded on purpose):
  1. extracted_pairs.json       -> from Django model `OshindongaWord` (confirmed Ndonga)
  2. autodict.py                -> a_words..w_z_words (~4000 entries, confirmed Ndonga spelling)
  3. new_words_one.py           -> new_words (~957 entries)
  4. dictionaryDatabase.py      -> nouns/verbs/adjectives/... (POS-tagged, ~1440 entries)
  5. phraseTranslation.py       -> myDictionary (~700 entries, includes conjugated phrases)
  6. CSV "Aa-ndonga" column     -> pronouns/kinship/core vocab (ground-truth checked)

Output:
  oshindonga_dictionary.json   -> {english: [{"osh": str, "pos": str|None, "source": str}]}
  needs_review.json            -> entries that looked like example sentences, not translations
"""
import ast, csv, json, re

UPLOADS = "/mnt/user-data/uploads/"
merged = {}       # english (lower, stripped) -> list of candidate dicts
needs_review = {} # english -> list of suspicious raw values + source

def add(english, osh, pos, source):
    if not english or not osh:
        return
    english = english.strip()
    osh = osh.strip()
    if not english or not osh:
        return
    key = english.lower()
    merged.setdefault(key, [])
    # de-dupe: same osh translation from same-ish meaning
    if not any(c["osh"].lower() == osh.lower() for c in merged[key]):
        merged[key].append({"osh": osh, "pos": pos, "source": source, "display_en": english})

def looks_like_example(text):
    """Heuristic: flag values that are usage examples, not translations."""
    if "=" in text:
        return True
    if len(text.split()) > 4 and text.strip().endswith((".", "!", "?")):
        return True
    return False

def flag(english, raw_value, source):
    needs_review.setdefault(english, []).append({"raw": raw_value, "source": source})

# ---------- 1. extracted_pairs.json ----------
with open(UPLOADS + "1788127672309_extracted_pairs.json", encoding="utf-8") as f:
    pairs = json.load(f)
POS_MAP = {"NN": "noun", "VB": "verb", "JJ": "adjective", "NNP": "proper_noun"}
for _id, row in pairs.items():
    if len(row) < 2:
        continue
    en, osh = row[0], row[1]
    pos = POS_MAP.get(row[3], None) if len(row) > 3 else None
    add(en, osh, pos, "extracted_pairs")

# ---------- 2. autodict.py ----------
src = open(UPLOADS + "1788127672308_autodict.py", encoding="utf-8").read()
idx = src.find("words = [n for n in d_k_words]")  # strip broken trailing block
if idx != -1:
    src = src[:idx]
ns = {}
exec(src, ns)
for varname, d in ns.items():
    if varname == "test_a" or varname.startswith("__") or not isinstance(d, dict):
        continue
    for en, vals in d.items():
        vals = vals if isinstance(vals, list) else [vals]
        for v in vals:
            if not isinstance(v, str):
                continue
            if looks_like_example(v):
                flag(en, v, "autodict")
            else:
                add(en, v, None, "autodict")

# ---------- 3. new_words_one.py ----------
src = open(UPLOADS + "1788127672309_new_words_one.py", encoding="utf-8").read()
ns = {}
exec(src, ns)
for en, vals in ns["new_words"].items():
    vals = vals if isinstance(vals, list) else [vals]
    for v in vals:
        if not isinstance(v, str):
            continue
        if looks_like_example(v):
            flag(en, v, "new_words_one")
        else:
            add(en, v, None, "new_words_one")

# ---------- 4. dictionaryDatabase.py (POS-tagged) ----------
src = open(UPLOADS + "1788127672308_dictionaryDatabase.py", encoding="utf-8").read()
ns = {}
exec(src, ns)
POS_VARS = {
    "nouns": "noun", "verbs": "verb", "adjectives": "adjective",
    "prepositions": "preposition", "adverbs": "adverb", "pronouns": "pronoun",
    "interjections": "interjection", "determiners": "determiner",
    "particles": "particle", "conjunctions": "conjunction", "phrases": "phrase",
}
for varname, pos in POS_VARS.items():
    d = ns.get(varname, {})
    for en, vals in d.items():
        vals = vals if isinstance(vals, list) else [vals]
        for v in vals:
            if not isinstance(v, str):
                continue
            if looks_like_example(v):
                flag(en, v, "dictionaryDatabase")
            else:
                add(en, v, pos, "dictionaryDatabase")

# ---------- 5. phraseTranslation.py (myDictionary) ----------
src = open(UPLOADS + "1788127672310_phraseTranslation.py", encoding="utf-8").read()
src = src[:src.find("def engToOsh")]  # only the dict literal
ns = {}
exec(src, ns)
for en, v in ns["myDictionary"].items():
    if isinstance(v, str):
        add(en, v, None, "phraseTranslation")

# ---------- 6. CSV Aa-ndonga column ----------
with open(UPLOADS + "1788127672310_Thesis_Dataset_-_Sheet_111__xlsx_-_Thesis_Dataset_-_Sheet_11_.csv",
          encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        en = row.get("English ", "").strip()
        osh = row.get("Aa-ndonga ", "").strip()
        if en and osh:
            add(en, osh, None, "csv_ndonga")

# ---------- Save ----------
with open("oshindonga_dictionary.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=1)

with open("needs_review.json", "w", encoding="utf-8") as f:
    json.dump(needs_review, f, ensure_ascii=False, indent=1)

print(f"Unique English entries: {len(merged)}")
print(f"Total translation candidates: {sum(len(v) for v in merged.values())}")
print(f"Entries flagged for manual review: {len(needs_review)}")
