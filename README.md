# English → Oshindonga Dictionary-Based Translator

A working v1 built from your existing data. No new data needed to run it.

## Files

| File | What it does |
|---|---|
| `build_dictionary.py` | Merges all your Oshindonga sources into one clean dictionary. Run this first (already run once — output included). |
| `oshindonga_dictionary.json` | The consolidated result: **5,424 unique English entries, 9,339 translation candidates.** |
| `needs_review.json` | **97 entries** that looked like example sentences mixed into the raw data (e.g. `"able" -> "The man is able to build his house="`) rather than real translations. These were kept OUT of the main dictionary automatically so they don't pollute output — but the real translations they contain are still buried in there and worth extracting by hand later. |
| `translator.py` | The actual translator (also runnable as a CLI). |

## What got merged, and from where

- `extracted_pairs.json` (your Django-exported "OshindongaWord" table) — 7,785 pairs, treated as the primary/most trustworthy source
- `dictionaryDatabase.py` — 1,438 entries, POS-tagged (noun/verb/adjective/etc.)
- `autodict.py` — 67 *new* entries after de-duplication (most of it already existed in `extracted_pairs.json`)
- `phraseTranslation.py` — 15 new entries
- Your CSV's **Aa-ndonga** column only — 35 new entries (pronouns/kinship terms)
- `new_words_one.py` — 0 new entries (fully redundant with `extracted_pairs.json`)
- **Excluded on purpose:** `oshindonga_data.json` — despite the filename, its data is actually labeled `oshikwanyama`, so it's the wrong dialect for what you asked for.

## How the translator works

1. Longest-match phrase substitution — it looks for the *longest* known English phrase first ("good morning" before "good"), so your 845 multi-word phrase entries actually get used instead of being shadowed by single-word matches.
2. Falls back to word-by-word when no phrase matches.
3. Anything with zero dictionary coverage is shown in `[brackets]` so you can see exactly what's missing.
4. When a word has multiple possible translations, it currently just prefers whichever came from your more-curated sources. It does **not** yet know which part of speech the sentence needs.

## Run it

```
cd oshindonga_translator
python3 translator.py
```

## The honest limitation (same one from before)

This is still word/phrase substitution, not grammar. Two known gaps, in priority order:

1. **Sense disambiguation.** "Water" has 3 candidate translations (noun *omeya* vs. verb forms *nwetha*/*tekela*) and right now it can guess wrong, like it did in testing (`"I want water"` → picked the verb form). Fixable next: a lightweight English POS tagger (e.g. NLTK) on the *input* sentence, so "water" after "want" is scored as a noun.
2. **Concord/agreement.** Oshindonga marks subject and noun-class agreement through prefixes that this system doesn't generate — it only recalls inflected forms that already exist as their own dictionary entries (like `"he threw" → "okwa umbu"`). No amount of dictionary size fixes this; it needs actual morphological rules layered on top, which is a bigger, separate project.

## Suggested next step

Want me to build the POS-disambiguation layer next (highest value, lowest effort — mostly fixes the "wrong sense picked" problem), or start on the needs_review.json cleanup so those ~97 buried translations get recovered into the main dictionary?
