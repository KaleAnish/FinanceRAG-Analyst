"""Extract readable text from SEC EDGAR primary documents (HTML / inline XBRL).

The research baseline used BeautifulSoup ``get_text(separator=" ")`` and then
collapsed every whitespace run, which destroyed paragraph boundaries. This
extractor keeps block structure so the sentence/paragraph segmenter in
``chunking.py`` can cut chunks on real boundaries:

* block elements (p, div, headings, list items, table rows) end with a blank line
* table cells are joined with `` | `` so financial tables stay row-readable
* hidden inline-XBRL headers, ``display:none`` blocks, scripts and styles are dropped

lxml is used directly because it is several times faster than BeautifulSoup on
multi-megabyte 10-K/10-Q documents.
"""

from __future__ import annotations

import re

from lxml import html as lxml_html

EXTRACTION_VERSION = "lxml_blocks_v1"

_DROP_TAGS = {"script", "style", "head", "title", "noscript", "ix:header"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "dl", "dt", "dd",
    "table", "tr", "pre", "center", "hr", "body", "center",
}
_CELL_TAGS = {"td", "th"}
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.IGNORECASE)

_CELL_SEPARATOR_RUN = re.compile(r"[ \t]*(?:\|[ \t]*){2,}")
_EDGE_PIPES = re.compile(r"^[ \t]*\|[ \t]*|[ \t]*\|[ \t]*$", re.MULTILINE)


def _tag_name(element) -> str:
    tag = element.tag
    if not isinstance(tag, str):  # comments / processing instructions
        return ""
    return tag.lower()


def extract_filing_text(raw: bytes | str) -> str:
    """Return block-structured plain text for one SEC HTML filing."""
    if isinstance(raw, bytes):
        if not raw.strip():
            return ""
        root = lxml_html.document_fromstring(raw)
    else:
        if not raw.strip():
            return ""
        root = lxml_html.document_fromstring(raw.encode("utf-8"))

    # Collect first, then remove: mutating while iterating skips elements.
    to_drop = []
    for element in root.iter():
        name = _tag_name(element)
        if not name:
            to_drop.append(element)
            continue
        if name in _DROP_TAGS:
            to_drop.append(element)
            continue
        style = element.get("style")
        if style and _HIDDEN_STYLE.search(style):
            to_drop.append(element)

    for element in to_drop:
        parent = element.getparent()
        if parent is None:
            continue
        # drop_tree keeps the element's tail text attached to the document.
        if hasattr(element, "drop_tree"):
            element.drop_tree()
        else:
            tail = element.tail
            parent.remove(element)
            if tail:
                parent.text = (parent.text or "") + tail

    for element in root.iter():
        name = _tag_name(element)
        if name in _BLOCK_TAGS:
            element.tail = "\n\n" + (element.tail or "")
        elif name in _CELL_TAGS:
            element.tail = " | " + (element.tail or "")
        elif name == "br":
            element.tail = "\n" + (element.tail or "")

    text = "".join(root.itertext())

    # Tidy table separators produced by empty cells and row edges.
    text = _CELL_SEPARATOR_RUN.sub(" | ", text)
    text = _EDGE_PIPES.sub("", text)
    return text
