import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import re
from re import Match

from mkdocs.plugins import get_plugin_logger
from openapidocs.mk.jinja import Jinja2DocumentsWriter
from openapidocs.mk.v3 import OpenAPIV3DocumentationHandler
from openapidocs.utils.source import read_from_source

log = get_plugin_logger("API-Docs-Generator")
rx = re.compile(r"\[OAD\(([^\)]+)\)\]")


def on_page_markdown(markdown: str, *args, **kwargs):
    if "[OAD(" in markdown:
        log.info("Found '[OAD' prefix. Parsing page...")
        return rx.sub(_replacer, markdown)
    return markdown


def _replacer(match: Match) -> str:
    source = match.group(1).strip("'\"")
    data = read_from_source(source)

    handler = OpenAPIV3DocumentationHandler(
        data, source=source, writer=Jinja2DocumentsWriter("theme")
    )
    log.info("Page parsed!")
    return handler.write()
