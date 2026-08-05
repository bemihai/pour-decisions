"""EPUB extraction backed by EbookLib and XHTML structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import ebooklib
from ebooklib import epub
from lxml import html

from .base import DocumentElement, DocumentExtractor, UnsupportedDocumentTypeError


_WHITESPACE_PATTERN = re.compile(r"\s+")
_CONTENT_XPATH = "//h1 | //h2 | //h3 | //p | //li | //table"


@dataclass
class _EpubContext:
    """Mutable XHTML heading context applied to emitted elements."""

    document_title: str = ""
    chapter: str = ""
    section: str = ""

    def apply_heading(self, text: str, level: int) -> None:
        """Update document context from an XHTML heading."""
        if level == 1:
            if not self.document_title:
                self.document_title = text
            elif _normalize_text(text).casefold() != _normalize_text(self.document_title).casefold():
                self.chapter = text
                self.section = ""
        elif level == 2:
            self.chapter = text
            self.section = ""
        else:
            self.section = text


class EbookLibExtractor(DocumentExtractor):
    """Extract ordered provider-neutral elements from an EPUB spine."""

    def extract(self, path: Path) -> list[DocumentElement]:
        """Extract XHTML headings and content in EPUB reading order."""
        source_path = Path(path)
        if source_path.suffix.lower() != ".epub":
            unsupported_suffix = source_path.suffix or "no suffix"
            raise UnsupportedDocumentTypeError(f"EbookLibExtractor does not support {unsupported_suffix}")

        book = epub.read_epub(str(source_path))
        context = _EpubContext(document_title=_book_title(book))
        elements: list[DocumentElement] = []
        for spine_index, item in enumerate(_ordered_document_items(book)):
            item_name = str(item.get_name() or "")
            for node in _content_nodes(item.get_body_content()):
                tag = _local_tag(node.tag)
                if _is_duplicate_nested_content(node, tag):
                    continue

                text = _normalize_text(" ".join(node.itertext()))
                if not text:
                    continue

                heading_level = int(tag[1]) if tag in {"h1", "h2", "h3"} else None
                if heading_level is not None:
                    context.apply_heading(text, heading_level)

                elements.append(
                    DocumentElement(
                        text=text,
                        source_path=str(source_path),
                        file_type="epub",
                        order_index=len(elements),
                        element_type=_element_type(tag),
                        heading_level=heading_level,
                        document_title=context.document_title,
                        chapter=context.chapter,
                        section=context.section,
                        metadata={
                            "spine_index": spine_index,
                            "item_name": item_name,
                        },
                    )
                )
        return elements


def _ordered_document_items(book: epub.EpubBook) -> Iterable[Any]:
    """Yield chapter documents in spine order, then any unreferenced chapters."""
    seen_ids: set[str] = set()
    for spine_entry in book.spine:
        item_id = str(spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry)
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT or not item.is_chapter():
            continue
        seen_ids.add(str(item.id))
        yield item

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if str(item.id) not in seen_ids and item.is_chapter():
            yield item


def _content_nodes(content: bytes) -> list[Any]:
    """Parse XHTML and return supported block nodes in document order."""
    if not content.strip():
        return []
    root = html.fromstring(content, parser=html.HTMLParser(encoding="utf-8"))
    return list(root.xpath(_CONTENT_XPATH))


def _is_duplicate_nested_content(node: Any, tag: str) -> bool:
    """Avoid emitting paragraph/list descendants already represented by a container."""
    ancestor_tags = {_local_tag(ancestor.tag) for ancestor in node.iterancestors()}
    if "table" in ancestor_tags and tag != "table":
        return True
    return tag == "p" and "li" in ancestor_tags


def _book_title(book: epub.EpubBook) -> str:
    """Return the first Dublin Core title when present."""
    titles = book.get_metadata("DC", "title")
    return _normalize_text(str(titles[0][0])) if titles else ""


def _element_type(tag: str) -> str:
    """Map supported XHTML tags to the normalized element vocabulary."""
    if tag in {"h1", "h2", "h3"}:
        return "heading"
    if tag == "li":
        return "list_item"
    if tag == "table":
        return "table"
    if tag == "p":
        return "paragraph"
    return "unknown"


def _local_tag(tag: Any) -> str:
    """Strip an optional XML namespace from a node tag."""
    return str(tag).rsplit("}", maxsplit=1)[-1].casefold()


def _normalize_text(text: str) -> str:
    """Collapse XHTML whitespace into normalized plain text."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()
