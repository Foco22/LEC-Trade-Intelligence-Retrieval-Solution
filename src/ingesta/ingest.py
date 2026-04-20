"""
LEC Trade Intelligence Corpus Downloader
=========================================
Descarga 1,000+ documentos relevantes para London Export Corporation:

  - GOV.UK Guidance API  → páginas de regulación de importación UK (Markdown)
  - WTO PDFs             → Trade Policy Reviews China + UK (PDF → Markdown)
  - FSA                  → Guidance de importación de bebidas (Markdown)

Todo se convierte a Markdown via MarkItDown para chunking estructurado.
No requiere API key. Todo es público y gratuito.

Uso:
    pip install requests tqdm markitdown[pdf]
    python ingesta.py

Estructura de salida:
    corps/
    ├── govuk/          → .txt con metadata en cabecera + cuerpo en Markdown
    ├── wto/            → .md convertidos desde PDF
    ├── fsa/            → .md con metadata en cabecera
    └── manifest.csv    → id, source, title, url, date, topic, filepath
"""

import asyncio
import csv
import hashlib
import os
import re
import tempfile
import time
import logging
from dataclasses import dataclass, fields, asdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from markitdown import MarkItDown
from dotenv import load_dotenv

from src.utils.mongodb import mongo
from src.utils.embeddings import embeddings

load_dotenv()

# ───────────────────────── CONFIG ─────────────────────────

OUTPUT_DIR   = Path("corps")
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"

SLEEP_BETWEEN_REQUESTS = 0.5   # segundos — sé amable con los servidores
REQUEST_TIMEOUT        = 30    # segundos por request

MD = MarkItDown()

# ── GOV.UK queries (no API key requerida)
GOVUK_QUERIES = [
    # Importación general China-UK
    {"q": "importing goods from China",          "topic": "china_import"},
    {"q": "import duty China UK",                "topic": "china_import"},
    {"q": "customs declaration import",          "topic": "customs"},
    {"q": "commodity codes import",              "topic": "customs"},
    {"q": "UK trade tariff guidance",            "topic": "tariff"},
    {"q": "Rules of origin import",              "topic": "tariff"},
    # Bebidas — LEC Beverages / Tsingtao
    {"q": "importing alcohol UK licence",        "topic": "beverages"},
    {"q": "alcohol duty import HMRC",            "topic": "beverages"},
    {"q": "food import regulations UK",          "topic": "beverages"},
    {"q": "beer import labelling UK",            "topic": "beverages"},
    {"q": "food standards import China",         "topic": "beverages"},
    # Robots / maquinaria — LEC Robotics
    {"q": "importing machinery UK regulations",  "topic": "robotics"},
    {"q": "UKCA marking import machinery",       "topic": "robotics"},
    {"q": "product safety import UK",            "topic": "robotics"},
    # Inversión / capital — LEC Global Capital
    {"q": "foreign direct investment UK China",  "topic": "investment"},
    {"q": "UK China trade agreement",            "topic": "trade_policy"},
    {"q": "sanctions export controls China",     "topic": "trade_policy"},
    {"q": "EORI number customs UK",              "topic": "customs"},
    {"q": "import VAT postponed accounting",     "topic": "customs"},
    {"q": "transfer pricing UK HMRC",            "topic": "investment"},
    # Bebidas — más ángulos
    {"q": "wine import licence UK",              "topic": "beverages"},
    {"q": "food contact materials import UK",    "topic": "beverages"},
    {"q": "nutritional labelling imported food", "topic": "beverages"},
    {"q": "alcohol personal licence UK",         "topic": "beverages"},
    {"q": "food safety imported products UK",    "topic": "beverages"},
    # Aduanas — más ángulos
    {"q": "customs warehouse UK guidance",       "topic": "customs"},
    {"q": "inward processing relief UK",         "topic": "customs"},
    {"q": "customs duty relief import",          "topic": "customs"},
    {"q": "UK global tariff commodity",          "topic": "tariff"},
    {"q": "anti-dumping duty China UK",          "topic": "tariff"},
    {"q": "tariff rate quotas UK imports",       "topic": "tariff"},
    {"q": "certificate of origin UK trade",      "topic": "tariff"},
    {"q": "import controls UK border force",     "topic": "customs"},
    {"q": "customs freight simplified procedure","topic": "customs"},
    # Robótica — más ángulos
    {"q": "electrical equipment safety import",  "topic": "robotics"},
    {"q": "electromagnetic compatibility UK",    "topic": "robotics"},
    {"q": "CE marking UKCA equivalence",         "topic": "robotics"},
    {"q": "industrial robots safety regulations","topic": "robotics"},
    # Inversión — más ángulos
    {"q": "UK company overseas investment",      "topic": "investment"},
    {"q": "VAT registration import business UK", "topic": "investment"},
    # Regulaciones laborales UK
    {"q": "UK employment law employer obligations",  "topic": "employment"},
    {"q": "minimum wage UK 2024",                    "topic": "employment"},
    {"q": "right to work UK checks employer",        "topic": "employment"},
    {"q": "UK visa work permit overseas employees",  "topic": "employment"},
    {"q": "PAYE payroll employer UK",                "topic": "employment"},
    {"q": "redundancy pay UK employer guide",        "topic": "employment"},
    {"q": "health and safety employer UK",           "topic": "employment"},
    {"q": "UK pension auto enrolment employer",      "topic": "employment"},
    {"q": "disciplinary grievance procedures UK",    "topic": "employment"},
    {"q": "UK employment tribunal guidance",         "topic": "employment"},
]

# ── WTO PDFs directos (patrón: s{número}_e.pdf)
WTO_PDFS = [
    # China Trade Policy Reviews
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s458_e.pdf",  "title": "WTO TPR China 2024",    "topic": "china_trade_policy"},
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s415_e.pdf",  "title": "WTO TPR China 2021",    "topic": "china_trade_policy"},
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s375_e.pdf",  "title": "WTO TPR China 2018",    "topic": "china_trade_policy"},
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s342_e.pdf",  "title": "WTO TPR China 2016",    "topic": "china_trade_policy"},
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s300_e.pdf",  "title": "WTO TPR China 2014",    "topic": "china_trade_policy"},
    # UK Trade Policy Reviews
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s430_e.pdf",  "title": "WTO TPR UK 2022",       "topic": "uk_trade_policy"},
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s385_e.pdf",  "title": "WTO TPR UK 2019",       "topic": "uk_trade_policy"},
    # Hong Kong (clave para LEC como gateway)
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s432_e.pdf",  "title": "WTO TPR Hong Kong 2022","topic": "hong_kong_trade"},
    # EU — contexto post-Brexit
    {"url": "https://www.wto.org/english/tratop_e/tpr_e/s426_e.pdf",  "title": "WTO TPR EU 2022",       "topic": "eu_trade_policy"},
]

# ── FSA páginas específicas (Food Standards Agency)
FSA_PAGES = [
    {"url": "https://www.food.gov.uk/business-guidance/importing-drinks",
     "title": "FSA Importing Drinks", "topic": "beverages"},
    {"url": "https://www.food.gov.uk/business-guidance/imports-exports",
     "title": "FSA Imports Exports Overview", "topic": "beverages"},
    {"url": "https://www.food.gov.uk/business-guidance/importing-food-from-outside-the-uk",
     "title": "FSA Importing Food from Outside UK", "topic": "beverages"},
    {"url": "https://www.food.gov.uk/business-guidance/food-labelling-and-packaging",
     "title": "FSA Food Labelling and Packaging", "topic": "beverages"},
    {"url": "https://www.food.gov.uk/business-guidance/approved-food-establishments",
     "title": "FSA Approved Establishments", "topic": "beverages"},
]

# ──────────────────────────────────────────────────────────


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class DocRecord:
    doc_id:   str
    source:   str   # govuk | wto | fsa
    title:    str
    url:      str
    date:     str
    topic:    str
    filepath: str
    checksum: str


# ─────────────────── HTTP SESSION ───────────────────

def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "LEC-RAG-Corpus-Builder/1.0 (research)"})
    return session


SESSION = make_session()


# ─────────────────── MANIFEST ───────────────────────

def load_manifest() -> set[str]:
    if not MANIFEST_CSV.exists():
        return set()
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        return {row["doc_id"] for row in csv.DictReader(f)}


def save_record(record: DocRecord) -> None:
    write_header = not MANIFEST_CSV.exists()
    with open(MANIFEST_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fd.name for fd in fields(DocRecord)])
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(record))


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(text: str, maxlen: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:maxlen].strip("_")


# ─────────────────── MARKITDOWN HELPERS ─────────────────

def html_to_md(html: str) -> str:
    """Convierte HTML a Markdown via MarkItDown usando un fichero temporal."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html",
                                     delete=False, encoding="utf-8") as f:
        f.write(html)
        tmp_path = f.name
    try:
        return MD.convert(tmp_path).text_content
    except Exception:
        # fallback: strip tags básico
        return re.sub(r"<[^>]+>", " ", html).strip()
    finally:
        os.unlink(tmp_path)


# ─────────────────── GOV.UK ─────────────────────────

GOVUK_API = "https://www.gov.uk/api/search.json"
GOVUK_CONTENT_API = "https://www.gov.uk/api/content"

def fetch_govuk(already_downloaded: set[str]) -> int:
    out_dir = OUTPUT_DIR / "govuk"
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    seen_urls: set[str] = set()

    for q_config in GOVUK_QUERIES:
        query = q_config["q"]
        topic = q_config["topic"]
        log.info(f"  GOV.UK query: '{query}'")

        params = {
            "q": query,
            "count": 50,
            "filter_content_store_document_type": ["guide", "detailed_guide",
                                                    "answer", "document_collection"],
        }

        try:
            resp = SESSION.get(GOVUK_API, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as e:
            log.warning(f"    search failed: {e}")
            time.sleep(2)
            continue

        for item in results:
            link   = item.get("link", "")
            title  = item.get("title", "no-title")
            date   = item.get("public_timestamp", "")[:10]
            full_url = f"https://www.gov.uk{link}"

            doc_id = "govuk_" + slug(link)

            if doc_id in already_downloaded or full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            content_url = f"{GOVUK_CONTENT_API}{link}"
            try:
                cr = SESSION.get(content_url, timeout=REQUEST_TIMEOUT)
                cr.raise_for_status()
                data = cr.json()
            except Exception as e:
                log.warning(f"    content fetch failed for {link}: {e}")
                time.sleep(1)
                continue

            text = extract_govuk_text(data, title, full_url, date, topic)
            if len(text.strip()) < 200:
                continue

            filename = out_dir / f"{doc_id}.txt"
            filename.write_text(text, encoding="utf-8")

            record = DocRecord(
                doc_id=doc_id,
                source="govuk",
                title=title,
                url=full_url,
                date=date,
                topic=topic,
                filepath=str(filename),
                checksum=checksum(filename),
            )
            save_record(record)
            already_downloaded.add(doc_id)
            downloaded += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        time.sleep(1)

    log.info(f"  GOV.UK: {downloaded} documentos descargados")
    return downloaded


def extract_govuk_text(data: dict, title: str, url: str,
                        date: str, topic: str) -> str:
    """Extrae texto del JSON de GOV.UK Content API; convierte HTML a Markdown."""
    lines = [
        f"TITLE: {title}",
        f"URL: {url}",
        f"DATE: {date}",
        f"TOPIC: {topic}",
        f"SOURCE: GOV.UK",
        "---",
    ]

    desc = data.get("description", "")
    if desc:
        lines.append(desc)
        lines.append("")

    details = data.get("details", {})

    for key in ("body", "introduction", "summary"):
        val = details.get(key, "")
        if isinstance(val, str) and val:
            lines.append(html_to_md(val))

    for part in details.get("parts", []):
        part_title = part.get("title", "")
        part_body  = part.get("body", "")
        if part_title:
            lines.append(f"\n## {part_title}")
        if isinstance(part_body, str) and part_body:
            lines.append(html_to_md(part_body))

    return "\n".join(lines)


# ─────────────────── WTO PDFs ───────────────────────

def fetch_wto(already_downloaded: set[str]) -> int:
    out_dir = OUTPUT_DIR / "wto"
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for item in WTO_PDFS:
        url   = item["url"]
        title = item["title"]
        topic = item["topic"]
        doc_id = "wto_" + slug(title)

        if doc_id in already_downloaded:
            log.info(f"  skip (ya existe): {title}")
            continue

        log.info(f"  WTO PDF: {title}")
        try:
            resp = SESSION.get(url, timeout=60, stream=True)
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"    failed: {e}")
            time.sleep(2)
            continue

        pdf_path = out_dir / f"{doc_id}.pdf"
        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)

        if pdf_path.stat().st_size < 10_000:
            log.warning(f"    archivo muy pequeño: {pdf_path}")
            pdf_path.unlink(missing_ok=True)
            continue

        # PDF → Markdown
        try:
            md_content = MD.convert(str(pdf_path)).text_content
        except Exception as e:
            log.warning(f"    MarkItDown conversion failed: {e}")
            pdf_path.unlink(missing_ok=True)
            continue

        md_path = out_dir / f"{doc_id}.md"
        md_path.write_text(md_content, encoding="utf-8")
        pdf_path.unlink(missing_ok=True)  # mantener solo .md

        record = DocRecord(
            doc_id=doc_id,
            source="wto",
            title=title,
            url=url,
            date="",
            topic=topic,
            filepath=str(md_path),
            checksum=checksum(md_path),
        )
        save_record(record)
        already_downloaded.add(doc_id)
        downloaded += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info(f"  WTO: {downloaded} PDFs convertidos a Markdown")
    return downloaded


# ─────────────────── FSA ────────────────────────────

def fetch_fsa(already_downloaded: set[str]) -> int:
    out_dir = OUTPUT_DIR / "fsa"
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for item in FSA_PAGES:
        url   = item["url"]
        title = item["title"]
        topic = item["topic"]
        doc_id = "fsa_" + slug(title)

        if doc_id in already_downloaded:
            log.info(f"  skip (ya existe): {title}")
            continue

        log.info(f"  FSA: {title}")
        try:
            md_content = MD.convert(url).text_content
        except Exception as e:
            log.warning(f"    failed: {e}")
            time.sleep(2)
            continue

        if len(md_content.strip()) < 200:
            continue

        header = "\n".join([
            f"TITLE: {title}",
            f"URL: {url}",
            f"DATE: ",
            f"TOPIC: {topic}",
            f"SOURCE: Food Standards Agency",
            "---",
            "",
        ])

        full_text = header + md_content
        filename  = out_dir / f"{doc_id}.md"
        filename.write_text(full_text, encoding="utf-8")

        record = DocRecord(
            doc_id=doc_id,
            source="fsa",
            title=title,
            url=url,
            date="",
            topic=topic,
            filepath=str(filename),
            checksum=checksum(filename),
        )
        save_record(record)
        already_downloaded.add(doc_id)
        downloaded += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info(f"  FSA: {downloaded} páginas convertidas a Markdown")
    return downloaded


# ─────────────────── CHUNKING ───────────────────────

MAX_CHUNK_CHARS = 1500  # ~400 tokens para bge-small-en-v1.5 (límite 512)


def chunk_document(text: str) -> list[str]:
    """Divide el texto en chunks por secciones ## y luego por párrafos si son muy largos."""
    sections = re.split(r"\n(?=##\s)", text)
    chunks = []

    for section in sections:
        if not section.strip():
            continue
        if len(section) <= MAX_CHUNK_CHARS:
            chunks.append(section.strip())
        else:
            paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 <= MAX_CHUNK_CHARS:
                    current = (current + "\n\n" + para).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = para
            if current:
                chunks.append(current)

    return chunks


# ─────────────────── INDEXADO ───────────────────────

async def index_corpus() -> None:
    """Lee el manifest, chunkea, embedea y guarda en MongoDB Atlas (incremental)."""
    if not MANIFEST_CSV.exists():
        log.warning("No se encontró manifest.csv — ejecuta la descarga primero.")
        return

    collection = mongo.get_collection()
    indexed = 0
    skipped = 0

    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        doc_id       = row["doc_id"]
        filepath     = row["filepath"]
        checksum_val = row["checksum"]

        # Incremental: saltar si ya está indexado con el mismo checksum
        existing = await collection.find_one(
            {"doc_id": doc_id, "metadata.checksum": checksum_val}
        )
        if existing:
            skipped += 1
            continue

        # Si el doc cambió, borrar chunks viejos
        await collection.delete_many({"doc_id": doc_id})

        try:
            text = Path(filepath).read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning(f"Archivo no encontrado: {filepath}")
            continue

        chunks = chunk_document(text)
        if not chunks:
            continue

        metadata = {
            "source":   row["source"],
            "title":    row["title"],
            "url":      row["url"],
            "date":     row["date"],
            "topic":    row["topic"],
            "checksum": checksum_val,
        }

        docs = []
        for i, chunk_text in enumerate(chunks):
            vector = embeddings.encode(chunk_text)
            docs.append({
                "doc_id":      doc_id,
                "chunk_index": i,
                "text":        chunk_text,
                "embedding":   vector,
                "metadata":    metadata,
            })

        await collection.insert_many(docs)
        indexed += 1
        log.info(f"  indexado: {doc_id} ({len(docs)} chunks)")

    log.info(f"Indexado completo — {indexed} docs nuevos, {skipped} sin cambios.")


# ─────────────────── MAIN ───────────────────────────

def print_summary():
    if not MANIFEST_CSV.exists():
        return
    counts: dict[str, int] = {}
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["source"]] = counts.get(row["source"], 0) + 1
    total = sum(counts.values())

    print("\n┌──────────────────────────────────┐")
    print("│      CORPUS LEC — RESUMEN        │")
    print("├──────────────────────────────────┤")
    for src, n in counts.items():
        bar = "█" * min(n // 5, 20)
        print(f"│  {src:<8}  {n:>4} docs  {bar:<20}│")
    print("├──────────────────────────────────┤")
    print(f"│  {'TOTAL':<8}  {total:>4} docs                  │")
    print("└──────────────────────────────────┘")
    print(f"\nManifest → {MANIFEST_CSV}")
    print("Listo para ingestar en el pipeline RAG ✓\n")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    already = load_manifest()

    if already:
        log.info(f"Incremental run — {len(already)} docs ya descargados, se saltarán.")

    print("\n🇬🇧  Descargando GOV.UK guidance...")
    fetch_govuk(already)

    print("\n🌍  Descargando WTO Trade Policy Reviews (PDF → Markdown)...")
    fetch_wto(already)

    print("\n🍺  Descargando FSA beverages guidance...")
    fetch_fsa(already)

    print_summary()

    print("\n🔍  Indexando en MongoDB Atlas...")
    asyncio.run(index_corpus())


if __name__ == "__main__":
    main()