"""Generate the code reference pages.

Source: https://mkdocstrings.github.io/recipes/#generate-pages-on-the-fly
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path(".").rglob("*.py")):
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path(".", doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    if (
        any([".venv" in part for part in parts])
        or parts[1] != "gptstonks-multiagents"
        or parts[2] == "tests"
    ):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with (
    mkdocs_gen_files.open("libs/gptstonks-multiagents/index.md", "w") as index_md,
    open("libs/gptstonks-multiagents/README.md") as readme_md,
):
    lines = [line.replace("../../docs/assets", "../../assets") for line in readme_md.readlines()]
    index_md.writelines(lines)
