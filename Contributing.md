🤝 Contributing to research-llm

Thank you for considering contributing to the research-llm project! This open-source project aims to build a complete research assistant pipeline using summarization, QLoRA, and RAG methods for institutional knowledge discovery.

🧭 Project Scope

The pipeline converts PDFs + metadata into a searchable vector database indexed by:

Researcher name

Paper title

Extracted summaries

Fine-tuned QA pairs

Final output supports chatbot-style question-answering using LLaMA + ChromaDB.

🛠 How to Contribute

1. Fork and Clone

git clone https://github.com/YOUR_USERNAME/research-llm.git
cd research-llm

2. Create a Branch

git checkout -b your-feature-name

3. Make Your Changes

Add your code or improvements

Write docstrings and comments

Ensure GPU acceleration is preserved (if applicable)

4. Format & Test

Ensure your code runs end-to-end using python run_pipeline.py

Check that major modules (PDF download, summarization, DB updates, QA, Chroma ingestion) are functioning properly

5. Commit & Push

git add .
git commit -m "Add <your feature>"
git push origin your-feature-name

6. Open a Pull Request

Go to your fork on GitHub and open a PR into the main branch.

✅ Contribution Guidelines

Code must be GPU-compatible and efficient

Write clear comments, especially around model or data logic

Prefer functions that can be unit tested in isolation

Use standard libraries unless necessary

Do not commit large model files (>100MB); use .gitignore

🙌 What You Can Contribute

🧠 Add new prompt templates for QA generation

🧪 Add unit tests for preprocessing or DB scripts

⚡ Improve tokenizer performance or masking logic

🔌 Plug in a chatbot frontend (e.g. Streamlit, FastAPI)

📜 Write documentation for each module

📩 Questions?

Open an issue or start a discussion on GitHub if you're not sure where to start or have questions about design decisions.

