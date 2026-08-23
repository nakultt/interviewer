from kec_runtime.resume import extract_resume


def test_plain_text_resume_extracts_known_skills() -> None:
    profile = extract_resume(
        "resume.txt",
        b"AI Engineer with five years of Python, FastAPI, MongoDB, RAG and MLX experience. "
        b"Built production retrieval systems and measured answer quality.",
    )
    assert "Python" in profile.skills
    assert "RAG" in profile.skills
    assert "MLX" in profile.skills
