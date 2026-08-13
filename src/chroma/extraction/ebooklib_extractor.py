"""EPUB extraction backed by EbookLib and XHTML structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable
from urllib.parse import unquote

import ebooklib
from ebooklib import epub
from lxml import html

from .base import DocumentElement, DocumentExtractor, UnsupportedDocumentTypeError
from ..structural_roles import classify_structural_role


_WHITESPACE_PATTERN = re.compile(r"\s+")
_CONTENT_XPATH = "//h1 | //h2 | //h3 | //p | //li | //table"
_ENTRY_CLASS_PATTERN = re.compile(r"(?:^|[-_])(?:entry|grape|variety)(?:$|[-_])", re.IGNORECASE)


@dataclass(frozen=True)
class _NavigationTarget:
    """One normalized EPUB navigation or table-of-contents target."""

    item_name: str
    anchor: str
    title: str
    depth: int


@dataclass
class _EpubContext:
    """Mutable XHTML heading context applied to emitted elements."""

    document_title: str = ""
    chapter: str = ""
    section: str = ""
    entry_title: str = ""
    entry_boundary_evidence: str = ""

    def apply_heading(self, text: str, level: int, *, entry_boundary_evidence: str = "") -> str:
        """Update context from a heading and return the rule that changed it."""
        normalized_text = _normalize_text(text)
        if level == 1 and not self.document_title:
            self.document_title = normalized_text
            return "document_title"
        if _normalize_text(self.document_title).casefold() == normalized_text.casefold():
            return "document_title_repeat"

        if entry_boundary_evidence:
            self.chapter = normalized_text
            self.section = ""
            self.entry_title = normalized_text
            self.entry_boundary_evidence = entry_boundary_evidence
            return entry_boundary_evidence

        if level == 1:
            self.chapter = normalized_text
            self.section = ""
            self.entry_title = ""
            self.entry_boundary_evidence = ""
            return "heading_level_1"
        if level == 2:
            if self.entry_title:
                self.section = normalized_text
                return "entry_subheading_level_2"
            self.chapter = normalized_text
            self.section = ""
            return "heading_level_2"

        self.section = normalized_text
        return "heading_level_3"


class EbookLibExtractor(DocumentExtractor):
    """Extract ordered provider-neutral elements from an EPUB spine."""

    def extract(self, path: Path) -> list[DocumentElement]:
        """Extract XHTML headings and content in EPUB reading order."""
        source_path = Path(path)
        if source_path.suffix.lower() != ".epub":
            unsupported_suffix = source_path.suffix or "no suffix"
            raise UnsupportedDocumentTypeError(f"EbookLibExtractor does not support {unsupported_suffix}")

        book = epub.read_epub(str(source_path))
        navigation_targets = _navigation_targets(book)
        context = _EpubContext(document_title=_book_title(book))
        elements: list[DocumentElement] = []
        for spine_index, item in enumerate(_ordered_document_items(book)):
            item_name = _normalize_item_name(str(item.get_name() or ""))
            nodes = _content_nodes(item.get_body_content())
            peer_entry_indexes = _peer_entry_heading_indexes(nodes)
            first_heading_seen = False
            for node_index, node in enumerate(nodes):
                tag = _local_tag(node.tag)
                if _is_duplicate_nested_content(node, tag):
                    continue

                text = _normalize_text(" ".join(node.itertext()))
                if not text:
                    continue

                heading_level = int(tag[1]) if tag in {"h1", "h2", "h3"} else None
                navigation_target = _navigation_target_for_node(
                    navigation_targets,
                    item_name=item_name,
                    node=node,
                    is_first_heading=heading_level is not None and not first_heading_seen,
                )
                entry_boundary_evidence = ""
                context_update_evidence = "inherited"
                if heading_level is not None:
                    entry_boundary_evidence = _entry_boundary_evidence(
                        node,
                        node_index=node_index,
                        peer_entry_indexes=peer_entry_indexes,
                        navigation_target=navigation_target,
                    )
                    context_update_evidence = context.apply_heading(
                        text,
                        heading_level,
                        entry_boundary_evidence=entry_boundary_evidence,
                    )
                    first_heading_seen = True

                node_ids = _node_anchor_ids(node)
                element_type = _element_type(tag)
                structural_role = classify_structural_role(
                    text,
                    {"chapter": context.chapter, "section": context.section},
                    element_type=element_type,
                ).role
                elements.append(
                    DocumentElement(
                        text=text,
                        source_path=str(source_path),
                        file_type="epub",
                        order_index=len(elements),
                        element_type=element_type,
                        heading_level=heading_level,
                        document_title=context.document_title,
                        chapter=context.chapter,
                        section=context.section,
                        structural_role=structural_role,
                        metadata={
                            "extraction_provider": "ebooklib",
                            "spine_index": spine_index,
                            "item_name": item_name,
                            "element_id": ",".join(node_ids),
                            "css_class": str(node.get("class") or ""),
                            "nav_title": navigation_target.title if navigation_target else "",
                            "nav_depth": navigation_target.depth if navigation_target else -1,
                            "entry_title": context.entry_title,
                            "entry_boundary": bool(entry_boundary_evidence),
                            "entry_boundary_evidence": context.entry_boundary_evidence,
                            "context_update_evidence": context_update_evidence,
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


def _navigation_targets(book: epub.EpubBook) -> dict[tuple[str, str], _NavigationTarget]:
    """Flatten EPUB navigation into item-and-anchor lookup keys."""
    targets: dict[tuple[str, str], _NavigationTarget] = {}

    def visit(entries: Iterable[Any], depth: int) -> None:
        for entry in entries:
            if isinstance(entry, tuple):
                section, children = entry
                _add_navigation_target(targets, section, depth)
                visit(children, depth + 1)
            else:
                _add_navigation_target(targets, entry, depth)

    visit(book.toc, 0)
    return targets


def _add_navigation_target(
    targets: dict[tuple[str, str], _NavigationTarget],
    entry: Any,
    depth: int,
) -> None:
    """Add one link-like navigation entry when it has a usable href."""
    href = str(getattr(entry, "href", "") or "")
    if not href:
        return
    item_name, separator, anchor = href.partition("#")
    normalized_item_name = _normalize_item_name(item_name)
    normalized_anchor = unquote(anchor) if separator else ""
    target = _NavigationTarget(
        item_name=normalized_item_name,
        anchor=normalized_anchor,
        title=_normalize_text(str(getattr(entry, "title", "") or "")),
        depth=depth,
    )
    targets[(target.item_name, target.anchor)] = target


def _navigation_target_for_node(
    targets: dict[tuple[str, str], _NavigationTarget],
    *,
    item_name: str,
    node: Any,
    is_first_heading: bool,
) -> _NavigationTarget | None:
    """Resolve a node against exact anchors, then an item-level first-heading target."""
    for anchor in _node_anchor_ids(node):
        target = targets.get((item_name, anchor))
        if target is not None:
            return target
    if is_first_heading:
        return targets.get((item_name, ""))
    return None


def _entry_boundary_evidence(
    node: Any,
    *,
    node_index: int,
    peer_entry_indexes: set[int],
    navigation_target: _NavigationTarget | None,
) -> str:
    """Return the strongest explicit reason that a heading begins a peer entry."""
    if navigation_target is not None:
        return "navigation_anchor"
    css_class = str(node.get("class") or "")
    if _looks_like_entry_class(css_class):
        return "css_entry_class"
    if node_index in peer_entry_indexes:
        return "peer_heading_run"
    return ""


def _looks_like_entry_class(css_class: str) -> bool:
    """Recognize explicit entry classes plus the inspected publisher's h2c marker."""
    class_tokens = {token.casefold() for token in css_class.split() if token.strip()}
    return "h2c" in class_tokens or any(_ENTRY_CLASS_PATTERN.search(token) for token in class_tokens)


def _peer_entry_heading_indexes(nodes: list[Any]) -> set[int]:
    """Find runs of short uppercase peer headings that represent dictionary-style entries."""
    heading_entries = [
        (index, _local_tag(node.tag), _normalize_text(" ".join(node.itertext())))
        for index, node in enumerate(nodes)
        if _local_tag(node.tag) in {"h1", "h2", "h3"}
    ]
    entry_indexes: set[int] = set()
    run: list[int] = []
    run_level = ""

    def flush() -> None:
        if len(run) >= 3:
            entry_indexes.update(run)
        run.clear()

    for node_index, tag, text in heading_entries:
        if _is_short_upper_heading(text) and (not run or tag == run_level):
            run.append(node_index)
            run_level = tag
            continue
        flush()
        if _is_short_upper_heading(text):
            run.append(node_index)
            run_level = tag
        else:
            run_level = ""
    flush()
    return entry_indexes


def _is_short_upper_heading(text: str) -> bool:
    """Return whether text has the shape of a compact uppercase dictionary entry."""
    return (
        bool(text)
        and len(text.split()) <= 4
        and len(text) <= 60
        and any(character.isalpha() for character in text)
        and text.isupper()
    )


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


def _node_anchor_ids(node: Any) -> list[str]:
    """Return stable IDs on a node or its inline anchor descendants."""
    ids: list[str] = []
    direct_id = str(node.get("id") or "").strip()
    if direct_id:
        ids.append(direct_id)
    for anchor in node.xpath(".//*[@id]"):
        anchor_id = str(anchor.get("id") or "").strip()
        if anchor_id and anchor_id not in ids:
            ids.append(anchor_id)
    return ids


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


def _normalize_item_name(item_name: str) -> str:
    """Normalize EPUB-internal paths for navigation matching."""
    normalized = unquote(item_name).replace("\\", "/").lstrip("/")
    return str(PurePosixPath(normalized)) if normalized else ""


def _local_tag(tag: Any) -> str:
    """Strip an optional XML namespace from a node tag."""
    return str(tag).rsplit("}", maxsplit=1)[-1].casefold()


def _normalize_text(text: str) -> str:
    """Collapse XHTML whitespace into normalized plain text."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()
