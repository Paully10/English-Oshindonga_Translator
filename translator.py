"""
Oshindonga <-> English dictionary-based translator (Ndonga only).

How it works:
  1. Load the consolidated dictionary (oshindonga_dictionary.json).
  2. Longest-match phrase substitution: scan the input for the LONGEST known
     English phrase/word first, replace it, then keep scanning the remainder.
     This is why multi-word entries ("good morning", "how are you") matter --
     they get caught before the translator falls back to word-by-word.
  3. Apply light Ndonga concord/agreement post-processing: subject/object
     pronoun-to-prefix agreement is NOT modeled here (that needs a real
     morphological analyzer -- see NOTE at bottom). What this layer does
     handle: choosing the right candidate when several translations exist
     for one English word, using simple POS-based ranking.
  4. Anything left with no dictionary match is returned in [BRACKETS] so you
     can see exactly what's missing -- useful for prioritizing what to add
     to the dictionary next.

Usage (CLI):
    python translator.py
    > Enter English text: Thank you, I am hungry
"""
import json
import re

DICT_PATH = "oshindonga_dictionary.json"


def load_dictionary(path=DICT_PATH):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # keep keys lowercase for matching, but remember original casing per entry
    return {k.lower(): v for k, v in raw.items()}


def pick_best(candidates, prefer_pos=None):
    """Choose one translation among several candidates for a word."""
    if not candidates:
        return None
    if prefer_pos:
        for c in candidates:
            if c.get("pos") == prefer_pos:
                return c["osh"]
    # Prefer entries that came from the POS-tagged source (dictionaryDatabase)
    # or the structured extracted_pairs source over looser word-list sources,
    # since those two were curated/reviewed more carefully.
    ranked = sorted(
        candidates,
        key=lambda c: 0 if c["source"] in ("dictionaryDatabase", "extracted_pairs") else 1,
    )
    return ranked[0]["osh"]


def tokenize_keep_punct(text):
    """Split into words + punctuation, preserving punctuation as separate tokens."""
    return re.findall(r"[A-Za-z']+|[^\sA-Za-z']", text)


def translate(text, dictionary, show_unmatched=True):
    """
    Longest-match substitution over the raw text (case-insensitive), phrase-first.
    Returns (translated_string, list_of_unmatched_spans).
    """
    lower = text.lower()
    n = len(lower)
    # sort keys longest-first so multi-word phrases are tried before single words
    keys_by_len = sorted(dictionary.keys(), key=len, reverse=True)

    result_tokens = []
    unmatched = []
    i = 0
    while i < n:
        if lower[i].isspace():
            i += 1
            continue
        matched = False
        for key in keys_by_len:
            klen = len(key)
            if klen == 0 or i + klen > n:
                continue
            if lower[i:i + klen] != key:
                continue
            # require a word boundary on both sides so "I" doesn't match inside "His"
            before_ok = i == 0 or not lower[i - 1].isalnum()
            after_ok = (i + klen == n) or not lower[i + klen].isalnum()
            if not (before_ok and after_ok):
                continue
            osh = pick_best(dictionary[key])
            if osh:
                result_tokens.append(osh)
                i += klen
                matched = True
                break
        if matched:
            continue
        # no dictionary match -- consume one "word" as unmatched, or one punctuation char
        m = re.match(r"[A-Za-z']+", lower[i:])
        if m:
            word = m.group(0)
            unmatched.append(word)
            result_tokens.append(f"[{word}]" if show_unmatched else word)
            i += len(word)
        else:
            result_tokens.append(text[i])
            i += 1

    return " ".join(result_tokens).replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!"), unmatched


def main():
    dictionary = load_dictionary()
    print(f"Loaded {len(dictionary)} English entries.")
    print("Type an English sentence to translate (or 'quit' to exit).\n")
    while True:
        text = input("Enter English text: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        translation, unmatched = translate(text, dictionary)
        print("Oshindonga:", translation)
        if unmatched:
            print(f"  (no dictionary match for: {', '.join(sorted(set(unmatched)))})")
        print()


if __name__ == "__main__":
    main()

# NOTE on grammar / concord:
# Oshindonga marks subject/object agreement and noun class through prefixes
# (e.g. "he is walking" vs "she is walking" often share the same pronoun root
# but differ in verb prefix depending on noun class). This dictionary-based
# engine does NOT generate those prefixes from grammar rules -- it only
# recalls whichever inflected form happens to already exist as its own
# dictionary entry (e.g. "he threw" -> "okwa umbu" is a literal lookup, not
# a generated inflection). This is why coverage of common conjugated phrases
# matters so much and is the main lever for improving output quality without
# more sentence-level training data.
