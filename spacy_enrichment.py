# SPACY NLP ENRICHMENT - Stage 3
# Extracts named entities (organizations, people, places) and cleaned
# noun-phrase keywords from article text. Used by the topic clustering
# stage to enrich each article and to help label topics.
#
# Requires the small English model:
#   python -m spacy download en_core_web_sm

import re
from functools import lru_cache

import spacy

# Entity labels we care about for risk signals.
#   ORG    - companies, agencies, institutions
#   PERSON - named individuals
#   GPE    - countries, cities, states (geo-political entities)
#   NORP   - nationalities, religious or political groups
#   LAW    - named laws / regulations
_ENTITY_LABELS = {"ORG", "PERSON", "GPE", "NORP", "LAW"}

_MODEL_NAME = "en_core_web_sm"

# Tokens that add noise to keywords / noun phrases.
_STOP_NOUN_CHUNKS = {
    "it", "he", "she", "they", "we", "i", "you", "this", "that", "these",
    "those", "who", "which", "what", "one", "some", "any", "all", "more",
    "the article", "the report", "the company", "the news",
}


@lru_cache(maxsize=1)
def _load_nlp():
    """Load and cache the spaCy pipeline.

    The parser is disabled (we only need the NER component plus the
    tagger for noun chunks) to keep processing fast on batches of
    articles. Falls back with a clear message if the model is missing.
    """
    try:
        # We keep the tagger + attribute ruler + lemmatizer for noun chunks,
        # and the NER. Disable nothing critical here; ner needs parser off is
        # fine because noun_chunks needs the parser -> keep parser enabled.
        return spacy.load(_MODEL_NAME)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{_MODEL_NAME}' is not installed. Run:\n"
            f"    python -m spacy download {_MODEL_NAME}"
        ) from exc


def _clean(text):
    """Normalize whitespace and strip control noise from a text field."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def _normalize_phrase(phrase):
    """Lowercase, strip, and drop leading determiners from a phrase."""
    phrase = phrase.lower().strip()
    phrase = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", phrase)
    return phrase.strip()


def enrich_text(text, nlp=None):
    """Run spaCy over a single text and return an enrichment dict.

    Returns a dict with:
      - entities: dict mapping entity label -> sorted list of unique texts
      - entity_str: flat ", "-joined string of all entity texts (for CSV)
      - keywords: list of cleaned noun-phrase keywords
      - keyword_str: ", "-joined keyword string (for CSV)
    """
    nlp = nlp or _load_nlp()
    text = _clean(text)

    empty = {
        "entities": {},
        "entity_str": "",
        "keywords": [],
        "keyword_str": "",
    }
    if not text:
        return empty

    doc = nlp(text)

    # --- named entities grouped by label ---
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in _ENTITY_LABELS:
            continue
        value = _clean(ent.text)
        if len(value) < 2:
            continue
        entities.setdefault(ent.label_, set()).add(value)

    entities = {label: sorted(vals) for label, vals in entities.items()}
    entity_str = ", ".join(
        v for label in sorted(entities) for v in entities[label]
    )

    # --- noun-phrase keywords ---
    keywords = []
    seen = set()
    for chunk in doc.noun_chunks:
        phrase = _normalize_phrase(chunk.text)
        # keep multi-char alphabetic phrases, drop pronouns / boilerplate
        if not phrase or phrase in _STOP_NOUN_CHUNKS:
            continue
        if not re.search(r"[a-z]", phrase):
            continue
        if len(phrase) < 3:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        keywords.append(phrase)

    return {
        "entities": entities,
        "entity_str": entity_str,
        "keywords": keywords,
        "keyword_str": ", ".join(keywords),
    }


def enrich_batch(texts, batch_size=64):
    """Efficiently enrich an iterable of texts using nlp.pipe.

    Yields one enrichment dict per input text, preserving order.
    """
    nlp = _load_nlp()
    cleaned = [_clean(t) for t in texts]

    for doc, original in zip(nlp.pipe(cleaned, batch_size=batch_size), cleaned):
        if not original:
            yield {"entities": {}, "entity_str": "", "keywords": [], "keyword_str": ""}
            continue

        entities = {}
        for ent in doc.ents:
            if ent.label_ not in _ENTITY_LABELS:
                continue
            value = _clean(ent.text)
            if len(value) < 2:
                continue
            entities.setdefault(ent.label_, set()).add(value)
        entities = {label: sorted(vals) for label, vals in entities.items()}
        entity_str = ", ".join(
            v for label in sorted(entities) for v in entities[label]
        )

        keywords = []
        seen = set()
        for chunk in doc.noun_chunks:
            phrase = _normalize_phrase(chunk.text)
            if not phrase or phrase in _STOP_NOUN_CHUNKS:
                continue
            if not re.search(r"[a-z]", phrase):
                continue
            if len(phrase) < 3:
                continue
            if phrase in seen:
                continue
            seen.add(phrase)
            keywords.append(phrase)

        yield {
            "entities": entities,
            "entity_str": entity_str,
            "keywords": keywords,
            "keyword_str": ", ".join(keywords),
        }


if __name__ == "__main__":
    # quick smoke test
    sample = (
        "The SEC fined a major bank over anti-money-laundering failures in "
        "New York, while regulators in the European Union weighed new rules."
    )
    result = enrich_text(sample)
    print("Entities:", result["entities"])
    print("Keywords:", result["keywords"])
