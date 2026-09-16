# Oshindonga Translator

A dictionary-based English ↔ Oshindonga translation system, built as the first step toward a full suite of machine translation tools for Namibia's local languages.

Oshindonga is one of the most widely spoken indigenous languages in Namibia, but like most Namibian languages it has almost no digital translation infrastructure and no meaningful parallel-sentence corpus available for training modern neural machine translation models. This project starts by solving the actual bottleneck — building and cleaning real language data — before attempting anything more ambitious.

## Status

**Working v1: dictionary-based translation.**

This is a rule-based system, not a trained AI model. It does not "learn" — it consists of a large consolidated dictionary and a longest-match substitution algorithm. This is a deliberate choice: with the current amount of parallel sentence data available for Oshindonga, a neural sequence-to-sequence model would perform worse than a well-built dictionary system, not better. See [Roadmap](#roadmap) for what comes next.

## What it does

- Translates English text to Oshindonga using longest-match phrase substitution — multi-word phrases ("good morning", "how are you") are matched as units before falling back to word-by-word translation.
- Flags any word or phrase with no dictionary coverage in `[brackets]`, so gaps in the dictionary are immediately visible.
- Runs as a simple interactive CLI.

## What it doesn't do (yet)

- **No grammar generation.** Oshindonga marks subject and noun-class agreement through prefixes. This system only recalls inflected forms that already exist as their own dictionary entries (e.g. `"he threw" → "okwa umbu"`); it does not generate agreement from rules.
- **No sense disambiguation.** When a word has multiple valid translations (e.g. "water" as a noun vs. a verb form), the system currently just prefers whichever source was more curated — not which sense fits the sentence.
- **No sentence-level fluency guarantees.** Output is understandable but not always grammatically correct Oshindonga.

## Project structure

```
oshindonga_translator/
├── build_dictionary.py        # Consolidates all raw data sources into one clean dictionary
├── translator.py               # The translation engine + CLI
├── oshindonga_dictionary.json  # Consolidated dictionary (5,424 English entries, ~9,300 translation candidates)
├── needs_review.json           # Entries auto-flagged as noisy (example sentences mixed into raw data)
└── README.md
```

## Data sources

The dictionary was consolidated from several original raw sources of varying quality and structure:

| Source | Contribution | Notes |
|---|---|---|
| `extracted_pairs.json` | 7,785 pairs | Primary/most trustworthy source — exported from a Django `OshindongaWord` model |
| `dictionaryDatabase.py` | 1,438 entries | POS-tagged (noun/verb/adjective/etc.) |
| `autodict.py` | 67 new entries | Mostly overlapping with `extracted_pairs.json` |
| CSV "Aa-ndonga" column | 35 new entries | Pronouns and kinship terms, ground-truth checked |
| `phraseTranslation.py` | 15 new entries | |
| `new_words_one.py` | 0 new entries | Fully redundant with `extracted_pairs.json` |

**Deliberately excluded:** a Kwanyama-language dataset that was mislabeled in its filename as Oshindonga — kept out to avoid mixing dialects.

## Setup

Requires Python 3.7+, no external dependencies.

```bash
git clone <repo-url>
cd oshindonga_translator
python3 build_dictionary.py   # only needed if you're re-running consolidation on updated source files
python3 translator.py
```

## Usage

```
$ python3 translator.py
Loaded 5424 English entries.
Type an English sentence to translate (or 'quit' to exit).

Enter English text: Thank you very much, I am hungry
Oshindonga: tangi unene [much], Onda sa ondjala.
```

Words in `[brackets]` have no dictionary match — useful for tracking exactly what vocabulary is still missing.

**Note:** if you're using this in an IDE, run it from an actual interactive terminal (not a read-only output/debug panel) — the CLI needs real stdin input.

## Roadmap

1. **Recover flagged entries** — `needs_review.json` holds ~97 entries where real translations were tangled up with example-sentence text in the raw data. Extracting them cleanly will grow the dictionary further.
2. **Part-of-speech disambiguation** — tag the input sentence to pick the right sense of ambiguous words (e.g. "water" the noun vs. the verb), the highest-leverage quality improvement available without new data.
3. **Morphological concord layer** — rule-based generation of Oshindonga's noun-class/subject agreement prefixes, to move beyond literal lookup toward actual grammar.
4. **Parallel sentence corpus collection** — sourcing real aligned sentence pairs (Bible translations, government documents, community contributions) to eventually support fine-tuning a pretrained multilingual model (e.g. NLLB-200) rather than training from scratch.
5. **Expand to other Namibian languages** — the pipeline (consolidate → clean → dictionary → translate) is not Oshindonga-specific. The project's source data already includes columns for Oshikwambi, Oshimbalanhu, Oshikwaluudhi, Oshikwanyama, Oshingandjera, and Oshimbandja, which map directly onto this same architecture.

## Contributing

The current highest-value contribution is data, not code: more verified English–Oshindonga sentence pairs directly unlock better translation quality (see item 4 in the roadmap). Dictionary corrections and additions to `oshindonga_dictionary.json` are also welcome.

## License

Build Fully by Paully Nampala
This is just the Beat, A larger model is on it's way, with all our Namibian indegenious languages!!!
