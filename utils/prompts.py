"""All prompt templates in one place."""
from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded research papers."

RAG_SYSTEM = f"""You are a careful research assistant. Answer the question using ONLY the context excerpts below.
Rules:
- Every excerpt starts with a label like [Document Name, Page X]. Cite the excerpts you rely on using exactly that label format.
- Never invent findings, numbers, or citations. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}"
- If the context only partly answers the question, answer the supported part and state what is missing.
- Be concise and precise."""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    ("human", "Conversation so far:\n{history}\n\nContext excerpts:\n{context}\n\nQuestion: {question}"),
])

SUMMARY_SECTIONS = [
    "Research Objective", "Problem Statement", "Proposed Methodology", "Dataset Used",
    "Key Findings", "Results", "Limitations", "Future Scope",
]

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You summarise research papers faithfully. Use ONLY the paper text provided. "
     "The text contains page markers like [Page 3]. If a section is not covered by the paper, "
     "write exactly 'Not mentioned in the paper.' Never invent facts, numbers or datasets."),
    ("human",
     "Paper: {document_name}\n\nPaper text:\n{paper_text}\n\n"
     "Write a structured summary in Markdown with exactly these headings (## level), in this order:\n"
     "{sections}\n\n"
     "Under each heading write 2-5 concise sentences or bullets. Cite pages like [{document_name}, Page X] "
     "where the claim comes from."),
])

COMPARE_DIMENSIONS = [
    "Research Objective", "Problem Statement", "Methodology", "Algorithms / Models",
    "Datasets", "Performance Metrics", "Key Results", "Limitations", "Research Gaps",
]

COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You compare research papers faithfully using ONLY the provided paper text. "
     "Page markers look like [Page 3]. If a paper does not cover a dimension, use 'Not mentioned'. "
     "Never invent facts. Respond with a single valid JSON object and nothing else (no code fences)."),
    ("human",
     "Papers:\n{papers_text}\n\n"
     "Return JSON with this exact shape:\n"
     '{{"table": {{"<dimension>": {{"<paper name>": "<short text with citation like [Paper Name, Page X]>"}}}}, '
     '"analysis": "<150-250 word comparative analysis in Markdown covering similarities, differences and research gaps>"}}\n'
     "Dimensions (use these exact keys): {dimensions}\n"
     "Paper names (use these exact keys): {paper_names}"),
])
