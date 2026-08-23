import io
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class ResumeProfile:
    filename: str
    text: str
    skills: tuple[str, ...]
    highlights: tuple[str, ...]


SKILLS = (
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "React",
    "Next.js",
    "FastAPI",
    "AWS",
    "Azure",
    "GCP",
    "Docker",
    "Kubernetes",
    "Terraform",
    "Ansible",
    "MongoDB",
    "PostgreSQL",
    "Redis",
    "Kafka",
    "Spark",
    "PyTorch",
    "TensorFlow",
    "MLX",
    "LLM",
    "RAG",
    "LangChain",
    "CI/CD",
    "Linux",
    "GitHub Actions",
)


def extract_resume(filename: str, content: bytes) -> ResumeProfile:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        document = Document(io.BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    elif suffix in {".txt", ".md"}:
        text = content.decode("utf-8", errors="replace")
    else:
        raise ValueError("Resume must be a PDF, DOCX, TXT, or Markdown file")
    clean = re.sub(r"[ \t]+", " ", text).strip()
    if len(clean) < 40:
        raise ValueError("The resume contains too little extractable text")
    found = tuple(skill for skill in SKILLS if re.search(rf"\b{re.escape(skill)}\b", clean, re.I))
    lines = tuple(line.strip(" •-\t") for line in clean.splitlines() if len(line.strip()) >= 35)
    highlights = lines[:8] if lines else (clean[:500],)
    return ResumeProfile(
        filename=filename, text=clean[:40_000], skills=found, highlights=highlights
    )
