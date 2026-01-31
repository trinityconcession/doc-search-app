# Doc Upload + AI Search (Streamlit + Dropbox)

## What it does
- Upload documents with dropdown: Invoice / Tax Exempt
- Conditional fields:
  - Invoice: Supplier + Amount
  - Tax Exempt: Customer + Business + Resale #
- Stores files in Dropbox (permanent)
- Extracts text from PDF/TXT/DOCX
- OCR for scanned PDFs/images (Streamlit Cloud uses packages.txt)
- AI semantic search (SentenceTransformers + FAISS)
- Download button for results

## Deploy
1) Push repo to GitHub
2) Deploy on Streamlit Community Cloud
3) Add Secrets:
   DROPBOX_ACCESS_TOKEN
   DROPBOX_FOLDER
