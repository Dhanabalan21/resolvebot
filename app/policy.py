"""Policy retrieval (the RAG part, assessment §2.4).

policies.md is split into documents at each top-level ``#`` heading, joined to
manifest.json by doc_id, and split again into numbered sections (§2, §2.1, ...).
Only documents that are ``current`` and already effective on APP_TODAY are used
as rules. Current documents that start later are returned separately as
"upcoming" so the assistant can mention a scheduled change, never apply it.

Search is a small BM25 keyword ranker with light stemming and synonyms; it runs
on a normal CPU and needs no extra dependencies.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("resolvebot.policy")

DOC_ID_RE = re.compile(r"Document ID:?\**\s*:?\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
HEADER_STATUS_RE = re.compile(r"Status:?\**\s*:?\s*([A-Za-z]+)", re.IGNORECASE)
SECTION_RE = re.compile(r"(?m)^(#{2,6})\s+(§\d+(?:\.\d+)*)\s*(.*?)\s*$")
HEADING_RE = re.compile(r"(?m)^#{2,6}\s")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "by", "with",
    "is", "are", "was", "be", "been", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "you", "your", "our", "us", "they", "them", "he", "she",
    "do", "does", "did", "can", "could", "would", "should", "will", "shall", "may",
    "what", "which", "who", "when", "where", "how", "why", "if", "then", "there",
    "about", "tell", "please", "know", "want", "need", "get", "have", "has", "had",
    "policy", "policies", "shopsphere", "any", "some", "much", "many",
    "s", "t", "under", "item", "items", "thing", "something", "anything", "one", "still",
}

# Query-side expansions: customer words -> words the policies use (stemmed).
SYNONYMS = {
    "earbud": ["earphon", "ear", "seal"],
    "earphon": ["earbud", "ear"],
    "headphon": ["earphon"],
    "open": ["seal", "unopen"],
    "unopen": ["seal", "open"],
    "seal": ["open"],
    "send": ["return"],
    "back": ["return"],
    "long": ["window", "time", "within"],
    "day": ["window", "within"],
    "deadlin": ["window"],
    "period": ["window", "month"],
    "cost": ["fee"],
    "pric": ["fee"],
    "charg": ["fee"],
    "fee": ["cost"],
    "deliveri": ["ship", "deliver"],
    "deliver": ["ship", "deliveri"],
    "money": ["refund"],
    "refund": ["money"],
    "broken": ["defectiv", "fault"],
    "faulti": ["defectiv", "fault"],
    "fault": ["defectiv"],
    "defect": ["defectiv", "fault"],
    "defectiv": ["fault"],
    "damag": ["defectiv"],
    "guarante": ["warranti"],
    "warranti": ["month"],
    "cancel": ["cancellation"],
    "cancellation": ["cancel"],
    "track": ["status", "eta"],
    "late": ["delay"],
    "lost": ["parcel"],
    "swap": ["exchang"],
    "human": ["support", "email"],
    "contact": ["support", "email"],
}

# Words that signal the customer is asking about a future/scheduled change.
CHANGE_HINTS = {"chang", "futur", "upcom", "new", "updat", "after", "next", "soon", "schedul"}


def _stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix in ("ations", "ation", "ables", "able", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("i" if suffix in ("ies", "ied") else "")
            break
    if word.endswith("y") and len(word) > 4:
        word = word[:-1] + "i"
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("₹", " inr "))
    return [_stem(w) for w in words if w not in STOP_WORDS]


@dataclass(frozen=True)
class Section:
    doc_id: str
    doc_title: str
    doc_type: str
    effective_date: date
    section: str
    title: str
    parent_title: str
    text: str
    tokens: tuple[str, ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    doc_type: str
    status: str
    effective_date: date


def _split_documents(markdown: str) -> dict[str, str]:
    blocks = re.split(r"(?m)^(?=#\s)", markdown)
    by_id: dict[str, str] = {}
    for block in blocks:
        match = DOC_ID_RE.search(block)
        if not match:
            continue
        doc_id = match.group(1).strip()
        if doc_id in by_id:
            logger.warning("Duplicate document %s in policies file; keeping the first", doc_id)
            continue
        by_id[doc_id] = block
    return by_id


def _extract_sections(doc: Document, block: str) -> list[Section]:
    sections: list[Section] = []
    parent_title = ""
    for heading in SECTION_RE.finditer(block):
        level = len(heading.group(1))
        number, title = heading.group(2), heading.group(3).strip()
        nxt = HEADING_RE.search(block, heading.end())
        body = block[heading.end(): nxt.start() if nxt else len(block)].strip()
        if level == 2:
            parent_title = title
        if not body:
            continue
        searchable = " ".join([doc.title, parent_title, title, title, body])
        sections.append(Section(
            doc_id=doc.doc_id,
            doc_title=doc.title,
            doc_type=doc.doc_type,
            effective_date=doc.effective_date,
            section=number,
            title=title,
            parent_title=parent_title if level > 2 else "",
            text=body,
            tokens=tuple(tokenize(searchable)),
        ))
    return sections


class PolicyIndex:
    """Parsed policy corpus. ``search`` filters by date at query time."""

    def __init__(self, documents: list[Document], sections: list[Section]):
        self.documents = {d.doc_id: d for d in documents}
        self.sections = sections
        self._position = {(s.doc_id, s.section): i for i, s in enumerate(sections)}
        self._title_tokens = {d.doc_id: set(tokenize(d.title)) for d in documents}
        n = max(len(sections), 1)
        df = Counter(t for s in sections for t in set(s.tokens))
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._avg_len = sum(len(s.tokens) for s in sections) / n

    @classmethod
    def load(cls, policies_path: Path, manifest_path: Path) -> "PolicyIndex":
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if isinstance(manifest, dict):  # tolerate {"documents": [...]}
            manifest = manifest.get("documents", [])
        blocks = _split_documents(Path(policies_path).read_text(encoding="utf-8"))

        documents, sections = [], []
        for entry in manifest:
            doc_id = str(entry.get("doc_id", "")).strip()
            block = blocks.get(doc_id)
            if not doc_id or block is None:
                logger.warning("Manifest entry %r has no matching document; skipped", doc_id)
                continue
            try:
                effective = date.fromisoformat(str(entry.get("effective_date", ""))[:10])
            except ValueError:
                logger.warning("Document %s has an invalid effective_date; skipped", doc_id)
                continue
            status = str(entry.get("status", "")).strip().lower()
            header = HEADER_STATUS_RE.search(block)
            if header and header.group(1).lower() == "superseded":
                status = "superseded"  # the stricter of manifest and header wins
            doc = Document(
                doc_id=doc_id,
                title=str(entry.get("title") or doc_id),
                doc_type=str(entry.get("doc_type", "policy")).strip().lower(),
                status=status,
                effective_date=effective,
            )
            documents.append(doc)
            sections.extend(_extract_sections(doc, block))
        return cls(documents, sections)

    # -- date filters -------------------------------------------------------
    def is_current(self, doc_id: str, today: date) -> bool:
        doc = self.documents.get(doc_id)
        return bool(doc and doc.status == "current" and doc.effective_date <= today)

    def is_upcoming(self, doc_id: str, today: date) -> bool:
        doc = self.documents.get(doc_id)
        return bool(doc and doc.status == "current" and doc.effective_date > today)

    def get_section(self, doc_id: str, section: str) -> Section | None:
        position = self._position.get((doc_id, section))
        return self.sections[position] if position is not None else None

    def has_section(self, doc_id: str, section: str, today: date) -> bool:
        """True if the section exists in a current or scheduled (not superseded) document."""
        return (doc_id, section) in self._position and (
            self.is_current(doc_id, today) or self.is_upcoming(doc_id, today)
        )

    # -- ranking ------------------------------------------------------------
    def _score(self, section: Section, weights: dict[str, float]) -> tuple[float, int]:
        tf = Counter(section.tokens)
        k1, b = 1.2, 0.75
        norm = k1 * (1 - b + b * len(section.tokens) / (self._avg_len or 1))
        score, hits = 0.0, 0
        for term, weight in weights.items():
            f = tf.get(term, 0)
            if not f:
                continue
            hits += 1
            score += weight * self._idf.get(term, 0) * f * (k1 + 1) / (f + norm)
        if any(t in self._title_tokens[section.doc_id] for t in weights):
            score *= 1.5  # the query names this document's topic
        if section.doc_type == "faq":
            score *= 0.6  # policy documents take precedence over the FAQ
        return score, hits

    def _ranked(self, weights: dict[str, float], keep) -> list[tuple[float, Section]]:
        scored = []
        for section in self.sections:
            if not keep(section.doc_id):
                continue
            score, hits = self._score(section, weights)
            if hits:  # at least one real query word must match
                scored.append((score, section))
        scored.sort(key=lambda pair: (-pair[0], pair[1].doc_type != "policy"))
        if not scored:
            return []
        top = scored[0][0]
        cutoff = max(top * 0.35, 0.5)
        kept = [pair for pair in scored if pair[0] >= cutoff]
        # Near-ties (broad questions like "what is the cancellation policy?") are
        # ordered as in the document, so the core rule (§1, §2) comes first.
        band = [pair for pair in kept if pair[0] >= top * 0.85]
        band.sort(key=lambda pair: self._position[(pair[1].doc_id, pair[1].section)])
        return band + kept[len(band):]

    def search(self, query: str, today: date, limit: int = 6) -> dict:
        base = tokenize(query)
        weights: dict[str, float] = {t: 1.0 for t in base}
        for term in base:
            for extra in SYNONYMS.get(term, []):
                weights.setdefault(extra, 0.5)

        selected: list[Section] = []
        per_doc: Counter = Counter()
        for _, section in self._ranked(weights, lambda d: self.is_current(d, today)):
            if per_doc[section.doc_id] >= 4:
                continue
            per_doc[section.doc_id] += 1
            selected.append(section)
            if len(selected) >= limit:
                break
        # Policy sections first: if the FAQ and a policy disagree, the policy applies.
        selected.sort(key=lambda s: s.doc_type != "policy")

        asks_about_change = bool(CHANGE_HINTS & set(base))
        if asks_about_change:
            # Scheduled-change notices are short: give the whole notice, in order.
            upcoming = [s for s in self.sections if self.is_upcoming(s.doc_id, today)][:limit]
        else:
            upcoming = [s for _, s in self._ranked(weights, lambda d: self.is_upcoming(d, today))][:3]
        if not asks_about_change and selected:
            # Only mention a scheduled change on the same topic as a current match.
            topics = {t for s in selected for t in tokenize(s.doc_title)}
            upcoming = [s for s in upcoming if topics & set(tokenize(s.doc_title))]

        return {
            "as_of": today.isoformat(),
            "matches": [self._as_result(s) for s in selected],
            "upcoming_changes": [
                {**self._as_result(s), "effective_date": s.effective_date.isoformat(),
                 "note": "Scheduled change, not in force yet. Do not apply it as a current rule."}
                for s in upcoming
            ],
        }

    @staticmethod
    def _as_result(section: Section) -> dict:
        heading = f"{section.parent_title} > {section.title}" if section.parent_title else section.title
        return {
            "doc_id": section.doc_id,
            "doc_type": section.doc_type,
            "section": section.section,
            "title": f"{section.doc_title}: {heading}",
            "text": section.text,
        }


@lru_cache(maxsize=4)
def _cached_index(policies_path: str, manifest_path: str, mtimes: tuple[float, float]) -> PolicyIndex:
    return PolicyIndex.load(Path(policies_path), Path(manifest_path))


def get_index(policies_path: Path, manifest_path: Path) -> PolicyIndex:
    """Index cached until either file changes on disk."""
    mtimes = (Path(policies_path).stat().st_mtime, Path(manifest_path).stat().st_mtime)
    return _cached_index(str(policies_path), str(manifest_path), mtimes)
