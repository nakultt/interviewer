from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoleProfile:
    id: str
    title: str
    summary: str
    competencies: tuple[str, ...]
    color: str


ROLES: tuple[RoleProfile, ...] = (
    RoleProfile(
        "ai-engineer",
        "AI Engineer",
        "Models, retrieval, evaluation, inference and responsible AI",
        ("machine learning", "LLM systems", "RAG", "evaluation", "Python", "MLOps"),
        "violet",
    ),
    RoleProfile(
        "cloud-ops-engineer",
        "Cloud Ops Engineer",
        "Cloud reliability, automation, observability and incident response",
        ("cloud architecture", "IaC", "Kubernetes", "observability", "SRE", "security"),
        "blue",
    ),
    RoleProfile(
        "backend-engineer",
        "Backend Engineer",
        "APIs, distributed systems, databases and production reliability",
        ("API design", "databases", "distributed systems", "testing", "performance"),
        "green",
    ),
    RoleProfile(
        "data-engineer",
        "Data Engineer",
        "Batch and streaming pipelines, warehouses, quality and governance",
        ("SQL", "data modeling", "ETL", "streaming", "data quality", "orchestration"),
        "orange",
    ),
    RoleProfile(
        "frontend-engineer",
        "Frontend Engineer",
        "Accessible interfaces, state, performance and frontend architecture",
        ("JavaScript", "React", "accessibility", "web performance", "testing"),
        "pink",
    ),
    RoleProfile(
        "devops-engineer",
        "DevOps Engineer",
        "Delivery pipelines, containers, automation and platform enablement",
        ("CI/CD", "containers", "IaC", "Linux", "security", "developer experience"),
        "cyan",
    ),
)


def get_role(role_id: str) -> RoleProfile:
    for role in ROLES:
        if role.id == role_id:
            return role
    raise KeyError(role_id)
