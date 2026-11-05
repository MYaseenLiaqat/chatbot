# Groq Document Chat

Upload one PDF, DOCX, or TXT file from the message composer, then ask questions
about it. The document remains available for follow-up questions during the
current chat session. A new upload replaces the previous document.

PDF text is extracted page by page with PyMuPDF. DOCX files are read paragraph
by paragraph, and TXT files are read as UTF-8. Scanned PDFs are reported as
unsupported because this basic example does not include OCR.

The PowerShell terminal prints labelled logs for upload, extraction, chunking,
retrieval, Groq requests, tool calls, and Groq responses so the RAG pipeline
and function-calling flow can be followed while learning.
