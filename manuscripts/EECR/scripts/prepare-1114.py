"""Download EECR originals and make a content-preserving, isolated CCDC copy."""
from __future__ import annotations

import concurrent.futures
import datetime
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
SUB = BASE / "submission/1114"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download(item: tuple[str, str, Path]) -> dict:
    label, url, target = item
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing source: {target}")
    proc = subprocess.run(
        ["curl.exe", "-L", "--max-time", "45", "--fail", "-sS", "-w", "\n%{http_code}\n%{url_effective}", url],
        capture_output=True,
    )
    parts = proc.stdout.rsplit(b"\n", 2)
    if proc.returncode or len(parts) != 3 or parts[1] != b"200":
        raise RuntimeError(f"Download failed: {url}; exit={proc.returncode}; {proc.stderr.decode(errors='replace')}")
    target.write_bytes(parts[0])
    record = {
        "label": label, "url": url, "effective_url": parts[2].decode(), "http_status": 200,
        "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "file": target.relative_to(BASE).as_posix(), "sha256": sha(target), "bytes": target.stat().st_size,
    }
    if target.suffix == ".html":
        soup = BeautifulSoup(parts[0], "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        body = soup.get_text("\n", strip=True)
        if len(body) < 100:
            raise ValueError(f"No usable page body at {url}")
        target.with_suffix(".txt").write_text(body, encoding="utf-8")
        record["links"] = [{"label": a.get_text(" ", strip=True), "href": a["href"]} for a in soup.find_all("a", href=True)]
    return record


def main() -> None:
    if (BASE / "latex/1114/main.tex").exists() or (SUB / "protected_before_hashes.json").exists():
        raise RuntimeError("Preparation already started; inspect existing evidence instead of overwriting it.")
    for name in ["latex/1114", "latex/template/downloads", "latex/template/official_latex", "images/1114", "output/1114/qa", "submission/1114/official_sources"]:
        (BASE / name).mkdir(parents=True, exist_ok=True)
    protected = {}
    for folder in [REPO / "manuscripts/CCDC", REPO / "manuscripts/ICACAR"]:
        for path in folder.rglob("*"):
            if path.is_file():
                protected[path.relative_to(REPO).as_posix()] = sha(path)
    save_json(SUB / "protected_before_hashes.json", protected)
    sources = SUB / "official_sources"
    downloads = BASE / "latex/template/downloads"
    requests = [(label, "https://www.eecr.org/" + leaf, sources / f"{label}.html") for label, leaf in [
        ("homepage", ""), ("submission", "sub.html"), ("call_for_papers", "cfp.html"),
        ("registration", "reg.html"), ("contact", "contact.html"), ("venue", "venue.html"), ("history_2026", "2026.html"),
    ]]
    requests += [
        ("official_latex_zip", "https://www.eecr.org/Conference-LaTeX-template.zip", downloads / "Conference-LaTeX-template.zip"),
        ("official_word_template", "https://www.eecr.org/template.docx", downloads / "template.docx"),
        ("official_cfp", "https://www.eecr.org/eecr_cfp.pdf", downloads / "eecr_cfp.pdf"),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(download, requests))
    save_json(sources / "index.json", [r for r in records if r["file"].endswith(".html")])
    save_json(BASE / "latex/template/download_manifest.json", [r for r in records if not r["file"].endswith(".html")])
    root = (BASE / "latex/template/official_latex").resolve()
    with zipfile.ZipFile(downloads / "Conference-LaTeX-template.zip") as archive:
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
                raise ValueError(f"Unsafe archive member: {name}")
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(name)
        archive.extractall(root)
    with zipfile.ZipFile(downloads / "template.docx") as archive:
        xml = ElementTree.fromstring(archive.read("word/document.xml"))
        text = "\n".join(e.text for e in xml.iter() if e.tag.split("}")[-1] == "t" and e.text)
        (BASE / "latex/template/template.docx.txt").write_text(text, encoding="utf-8")
    official_class = next(root.rglob("IEEEtran.cls"))
    source = REPO / "manuscripts/CCDC/latex/1029"
    assert sha(official_class) == sha(source / "IEEEtran.cls"), "Official class differs; review before migrating."
    original = (source / "main.tex").read_bytes()
    migrated = original.replace(
        b"% CCDC official conference format. Do not override its layout or typography.",
        b"% EECR official IEEEtran conference format. [ADAPTED: venue annotation only]",
    ).replace(b"../../images/1029/", b"../../images/1114/")
    (BASE / "latex/1114/main.tex").write_bytes(migrated)
    shutil.copyfile(source / "references.bib", BASE / "latex/1114/references.bib")
    shutil.copyfile(official_class, BASE / "latex/1114/IEEEtran.cls")
    assets = []
    for name in re.findall(rb"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", original):
        src = (source / name.decode()).resolve()
        dst = BASE / "images/1114" / src.name
        shutil.copyfile(src, dst)
        assets.append({"source": src.relative_to(REPO).as_posix(), "target": dst.relative_to(REPO).as_posix(), "sha256": sha(src)})
    save_json(SUB / "asset_manifest.json", assets)
    diff = "".join(difflib.unified_diff(original.decode().splitlines(True), migrated.decode().splitlines(True), fromfile="CCDC/latex/1029/main.tex", tofile="EECR/latex/1114/main.tex"))
    (SUB / "migration_source.diff").write_text(diff, encoding="utf-8")
    compared = {}
    for venue in ["CCDC", "ICACAR"]:
        compared[venue] = {"class_sha256_equal": sha(official_class) == sha(REPO / f"manuscripts/{venue}/latex/{'1029' if venue == 'CCDC' else '1114'}/IEEEtran.cls")}
    save_json(BASE / "latex/template/template_comparison.json", {"official_class": official_class.relative_to(BASE).as_posix(), "sha256": sha(official_class), "comparisons": compared})
    print(json.dumps({"downloaded_sources": len(records), "protected_files": len(protected), "images": len(assets), "class_comparison": compared}, indent=2))


if __name__ == "__main__":
    main()
