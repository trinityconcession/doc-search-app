import re
import io
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import streamlit as st

from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss

# OCR deps (on Streamlit Cloud these will work if packages.txt is included)
OCR_AVAILABLE = True
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_bytes
except Exception:
    OCR_AVAILABLE = False

# Dropbox
import dropbox


# =========================
# Config / Paths
# =========================
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "docs.db"
FAISS_PATH = DATA_DIR / "faiss.index"
MAP_PATH = DATA_DIR / "faiss_doc_ids.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOC_TYPES = ["Invoice", "Tax Exempt"]


# =========================
# Dropbox Storage (Files only)
# =========================
def get_dropbox_client() -> dropbox.Dropbox:
    token = st.secrets["DROPBOX_ACCESS_TOKEN"]
    return dropbox.Dropbox(token)

def dropbox_folder() -> str:
    # Safe default: app folder (Dropbox app created with "App folder" access)
    return st.secrets.get("DROPBOX_FOLDER", "/Apps/streamlit-doc-app").rstrip("/")

def dropbox_save(filename: str, content: bytes) -> str:
    """
    Saves file to Dropbox and returns Dropbox path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = filename.replace("/", "_").replace("\\", "_")
    dbx_path = f"{dropbox_folder()}/{ts}__{safe}"

    dbx = get_dropbox_client()
    dbx.files_upload(content, dbx_path, mode=dropbox.files.WriteMode.overwrite)
    return dbx_path

def dropbox_load(dbx_path: str) -> bytes:
    dbx = get_dropbox_client()
    md, res = dbx.files_download(dbx_path)
    return res.content


# =========================
# SQLite DB (Metadata + extracted text)
# NOTE: On Streamlit Cloud, local app storage can reset on restart.
# Dropbox keeps files permanent. Metadata may reset unless you add external DB later.
# =========================
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,

            supplier_name TEXT,
            amount REAL,

            customer_name TEXT,
            business_name TEXT,
            resale_number TEXT,

            filename TEXT NOT NULL,
            dropbox_path TEXT NOT NULL,

            extracted_text TEXT,
            uploaded_at TEXT NOT NULL
        );
        """)
init_db()

def insert_doc(row: dict) -> int:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("""
            INSERT INTO documents
            (doc_type, supplier_name, amount, customer_name, business_name, resale_number,
             filename, dropbox_path, extracted_text, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["doc_type"],
            row.get("supplier_name"),
            row.get("amount"),
            row.get("customer_name"),
            row.get("business_name"),
            row.get("resale_number"),
            row["filename"],
            row["dropbox_path"],
            row.get("extracted_text"),
            row["uploaded_at"],
        ))
        con.commit()
        return int(cur.lastrowid)

def get_doc_by_id(doc_id: int):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        return con.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

def fetch_all_docs_for_rebuild():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        return con.execute("SELECT id, extracted_text FROM documents ORDER BY id ASC").fetchall()


# =========================
# Text extraction + OCR
# =========================
def clean_text(t: str) -> str:
    t = t or ""
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_text_pdf_bytes(pdf_bytes: bytes) -> str:
    parts = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            parts.append(page.extract_text() or "")
    except Exception:
        return ""
    return "\n".join(parts)

def extract_text_docx_bytes(docx_bytes: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(docx_bytes))
        return "\n".join(p.text for p in d.paragraphs if p.text)
    except Exception:
        return ""

def ocr_image_bytes(img_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""

def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = 6) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=max_pages)
        texts = []
        for img in images:
            texts.append(pytesseract.image_to_string(img) or "")
        return "\n".join(texts)
    except Exception:
        return ""

def extract_all_text(filename: str, file_bytes: bytes) -> Tuple[str, str]:
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        try:
            return file_bytes.decode("utf-8", errors="ignore"), "txt"
        except Exception:
            return "", "none"

    if ext == ".docx":
        t = extract_text_docx_bytes(file_bytes)
        return t, "docx_text" if t.strip() else "none"

    if ext == ".pdf":
        t = extract_text_pdf_bytes(file_bytes)
        if t.strip():
            return t, "pdf_text"
        t2 = ocr_pdf_bytes(file_bytes)
        return t2, "ocr_pdf" if t2.strip() else "none"

    if ext in [".png", ".jpg", ".jpeg"]:
        t = ocr_image_bytes(file_bytes)
        return t, "ocr_image" if t.strip() else "none"

    return "", "none"


# =========================
# Embeddings + FAISS
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def load_or_create_index(dim: int):
    if FAISS_PATH.exists():
        return faiss.read_index(str(FAISS_PATH))
    return faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors

def save_index(idx):
    faiss.write_index(idx, str(FAISS_PATH))

def load_mapping():
    if MAP_PATH.exists():
        return np.load(MAP_PATH).tolist()
    return []

def save_mapping(doc_ids):
    np.save(MAP_PATH, np.array(doc_ids, dtype=np.int64))

def embed_text(model, text: str) -> np.ndarray:
    v = model.encode([text], normalize_embeddings=True)
    return v.astype("float32")

def rebuild_index_from_db(model):
    rows = fetch_all_docs_for_rebuild()

    idx = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    doc_ids = []
    vectors = []

    for r in rows:
        t = clean_text(r["extracted_text"] or "")
        if not t:
            continue
        vectors.append(model.encode([t], normalize_embeddings=True)[0])
        doc_ids.append(int(r["id"]))

    if vectors:
        vecs = np.array(vectors, dtype="float32")
        idx.add(vecs)

    save_index(idx)
    save_mapping(doc_ids)
    return idx, doc_ids


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Doc Upload + AI Search (Dropbox)", layout="wide")
st.title("Document Upload + AI Search (Dropbox Storage)")

# reset uploader trick
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Load model + index
model = load_model()
dim = model.get_sentence_embedding_dimension()

index = load_or_create_index(dim)
doc_id_map = load_mapping()

# Keep FAISS + mapping consistent
if index.ntotal != len(doc_id_map):
    index, doc_id_map = rebuild_index_from_db(model)

page = st.sidebar.radio("Menu", ["Upload", "AI Search"], index=0)

# ---------------- Upload ----------------
if page == "Upload":
    st.subheader("Upload")

    with st.form("upload_form", clear_on_submit=True):
        doc_type = st.selectbox("Document Type", DOC_TYPES)

        supplier_name = None
        amount = None
        customer_name = None
        business_name = None
        resale_number = None

        if doc_type == "Invoice":
            supplier_name = st.text_input("Supplier Name *")
            amount = st.number_input("Amount *", min_value=0.0, step=0.01, format="%.2f")
        else:
            customer_name = st.text_input("Customer Name *")
            business_name = st.text_input("Business Name *")
            resale_number = st.text_input("Resale Number *")

        uploaded = st.file_uploader(
            "Attach File * (pdf/txt/docx; images searchable with OCR)",
            type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
            key=f"file_{st.session_state.uploader_key}"
        )

        submitted = st.form_submit_button("Submit")

    if submitted:
        errors = []
        if doc_type == "Invoice":
            if not (supplier_name or "").strip():
                errors.append("Supplier Name is required for Invoice.")
            if amount is None or amount <= 0:
                errors.append("Amount must be greater than 0 for Invoice.")
        else:
            if not (customer_name or "").strip():
                errors.append("Customer Name is required for Tax Exempt.")
            if not (business_name or "").strip():
                errors.append("Business Name is required for Tax Exempt.")
            if not (resale_number or "").strip():
                errors.append("Resale Number is required for Tax Exempt.")

        if uploaded is None:
            errors.append("File attachment is required.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            file_bytes = uploaded.getvalue()

            # 1) Save file to Dropbox (permanent)
            dbx_path = dropbox_save(uploaded.name, file_bytes)

            # 2) Extract text (OCR fallback)
            extracted, method = extract_all_text(uploaded.name, file_bytes)
            extracted = clean_text(extracted)

            # 3) Save metadata to SQLite
            doc_id = insert_doc({
                "doc_type": doc_type,
                "supplier_name": supplier_name.strip() if supplier_name else None,
                "amount": float(amount) if doc_type == "Invoice" else None,
                "customer_name": customer_name.strip() if customer_name else None,
                "business_name": business_name.strip() if business_name else None,
                "resale_number": resale_number.strip() if resale_number else None,
                "filename": uploaded.name,
                "dropbox_path": dbx_path,
                "extracted_text": extracted,
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            })

            # 4) Add to FAISS index for AI search
            if extracted:
                v = embed_text(model, extracted)
                index.add(v)
                doc_id_map.append(doc_id)
                save_index(index)
                save_mapping(doc_id_map)
                st.success(f"Saved to Dropbox + indexed! (extraction: {method})")
            else:
                st.warning(
                    f"Saved to Dropbox, but no text extracted (extraction: {method}). "
                    "If it’s scanned PDF/image, OCR must be available."
                )

            # 5) Refresh/clear form and uploader
            st.session_state.uploader_key += 1
            st.rerun()

    if not OCR_AVAILABLE:
        st.info("OCR not available in this environment. PDFs with text still work; scanned docs need OCR.")


# ---------------- AI Search ----------------
else:
    st.subheader("AI Semantic Search")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Search (example: 'invoice from sysco 1200' or 'tax exempt resale number')")

    with c2:
        doc_type_filter = st.selectbox("Type filter", ["All"] + DOC_TYPES, index=0)

    with c3:
        top_k = st.selectbox("Top results", [5, 10, 20, 50], index=1)

    if st.button("Search", disabled=not bool(q.strip())):
        if index.ntotal == 0:
            st.warning("No indexed documents yet. Upload documents first.")
        else:
            qv = embed_text(model, clean_text(q))
            D, I = index.search(qv, int(top_k))

            shown = 0
            for score, pos in zip(D[0], I[0]):
                if pos < 0 or pos >= len(doc_id_map):
                    continue

                doc_id = int(doc_id_map[pos])
                r = get_doc_by_id(doc_id)
                if not r:
                    continue
                if doc_type_filter != "All" and r["doc_type"] != doc_type_filter:
                    continue

                shown += 1
                st.markdown(f"### #{r['id']} • {r['doc_type']} • {r['filename']}")
                cols = st.columns(4)

                if r["doc_type"] == "Invoice":
                    cols[0].write(f"**Supplier:** {r['supplier_name'] or '-'}")
                    cols[1].write(f"**Amount:** {r['amount'] if r['amount'] is not None else '-'}")
                else:
                    cols[0].write(f"**Customer:** {r['customer_name'] or '-'}")
                    cols[1].write(f"**Business:** {r['business_name'] or '-'}")
                    cols[2].write(f"**Resale #:** {r['resale_number'] or '-'}")

                cols[3].write(f"**Score:** {float(score):.3f}")

                preview = (r["extracted_text"] or "")[:350]
                if preview:
                    st.caption(preview)

                # Download from Dropbox
                file_bytes = dropbox_load(r["dropbox_path"])
                st.download_button(
                    label="Download file",
                    data=file_bytes,
                    file_name=r["filename"],
                    mime="application/octet-stream",
                    key=f"dl_{r['id']}"
                )

                st.code(r["dropbox_path"], language="text")
                st.divider()

            st.info(f"Displayed {shown} result(s).")
