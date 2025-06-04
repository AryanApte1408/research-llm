🧠 research-llm: Complete README

This document provides a complete overview of the research-llm pipeline, including the purpose of the project, its structure, and an in-depth explanation of every file and function.

📖 What is research-llm?

research-llm is a comprehensive Retrieval-Augmented Generation (RAG) pipeline developed to convert research papers affiliated with Syracuse University into a searchable knowledge base. It uses NLP techniques (summarization, QLoRA fine-tuning) and vector databases (ChromaDB) to enable a chatbot to answer user queries based on institutional research.

📊 Project Structure (Visual Overview)

graph TD
    A[CSV files + PDF URLs] --> B[pdfs.py → download_pdfs/]
    B --> C[ingest_pdf_fulltext.py → works table]
    B --> D[ingest_pdf_metadata() + csv_handler.py → research_info table]
    D --> E[clean_db.py → normalize names/titles]
    C --> F[summarize_works.py → T5 model → summaries in works]
    D --> G[llama_data_formatter.py → QA pairs]
    G --> H[fine_tune_llama_rag.py → fine-tuned LLaMA (QLoRA)]
    E --> I[migrate_to_chromadb.py → ChromaDB: paper_metadata]
    F --> J[migrate_to_chromadb.py → ChromaDB: paper_summaries]

🔍 File-by-File, Function-by-Function Breakdown

run_pipeline.py

Purpose: Orchestrates the end-to-end pipeline.

main(): Executes every stage in order, handling exceptions and cleanup.

ingest_pdf_metadata(): Inserts extracted metadata from PDFs into the research_info table.

pdfs.py

Purpose: Downloads research papers from pdf_url in CSV files.

main(): Iterates through rows, fetches and saves PDFs using requests, handles errors.

ingest_pdf_fulltext.py

Purpose: Reads PDF content and stores it into the works table.

main(): Processes all files in download_pdfs, skips corrupt/duplicate files.

work_exists(file_name, conn): Checks for duplicate entries.

csv_handler.py

Purpose: Reads multiple CSVs and loads metadata into research_info.

combine_csvs(): Normalizes column names and merges multiple CSVs.

populate_research_info_from_csv(): Inserts the cleaned metadata into the database.

clean_db.py

Purpose: Cleans junk data in research_info using parsed full text.

clean(txt): Strips non-ASCII and whitespace.

looks_dummy_title() / looks_dummy_name(): Identifies poor quality data.

parse_from_fulltext(txt): Extracts structured fields (title, DOI, date).

main(): Iterates through rows, applies fixes, and updates the database.

database_handler.py

Purpose: Helper functions for SQLite access.

insert_work(file_name, full_text): Adds a row to works.

fetch_unsummarized_works(limit): Selects works needing summaries.

update_summary(work_id, summary): Updates summary and status.

close_connection(): Safe placeholder.

pdf_pre.py

Purpose: Extracts raw text and metadata from a PDF.

extract_raw_text_from_pdf(): Reads all pages via PyPDF2.

clean_text(): Cleans up formatting.

extract_research_info_from_pdf(): Extracts title/authors/info heuristically.

model.py

Purpose: Loads or fine-tunes a t5-small model for summarization.

load_t5_model(): Caches model/tokenizer.

clear_memory(): Clears GPU memory.

summarize_text(text): Summarizes a paper.

fine_tune_t5_on_papers(df, output_dir): Fine-tunes T5 and saves artifacts.

summarize_works.py

Purpose: Applies T5 to all unsummarized entries.

main(limit): Loops through all works entries and updates summary field.

llama_data_formatter.py

Purpose: Builds question-answer pairs from metadata.

generate_qa_pairs(): Generates 4 QA entries per row; stores as pickle.

fine_tune_llama_rag.py

Purpose: Fine-tunes 4-bit LLaMA using LoRA on QA pairs.

load_llama_model(): Loads quantized model + tokenizer.

fine_tune_llama_on_papers(df): Prepares dataset, masks prompts, trains and saves model.

migrate_to_chromadb.py

Purpose: Inserts all summaries + metadata into ChromaDB collections.

_safe(x): Cleans null values.

migrate_metadata(): Converts research_info into Chroma documents.

migrate_summaries(): Converts works.summary into Chroma documents.

✅ Summary

This project represents a robust AI-powered research assistant pipeline that combines:

PDF ingestion

NLP summarization

QA generation

Language model fine-tuning

Vector database search

It enables fast, contextual querying of Syracuse University research output using modern LLMs.

⚙️ Requirements

Install the required dependencies:

pip install -r requirements.txt

Key libraries used:

torch

transformers

datasets

peft

bitsandbytes

chromadb

pandas, numpy, sqlite3, requests, PyPDF2, tqdm

Ensure CUDA-enabled GPU is available for model training/inference.

▶️ How to Run

Prepare your environment:

Place all source CSV files in ~/Downloads/Application/

Ensure T5 and LLaMA models are accessible locally

Run the full pipeline:

python run_pipeline.py

This will:

Download PDFs

Extract full-text

Populate metadata tables

Summarize with T5

Generate QA data

Fine-tune LLaMA using QLoRA

Populate ChromaDB
