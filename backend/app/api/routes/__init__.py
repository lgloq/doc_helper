from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.eval import router as eval_router
from app.api.routes.faqs import router as faqs_router
from app.api.routes.health import router as health_router
from app.api.routes.observability import router as observability_router
from app.api.routes.reports import router as reports_router
from app.api.routes.search import router as search_router
from app.api.routes.tasks import router as tasks_router

__all__ = [
    "auth_router",
    "chat_router",
    "documents_router",
    "eval_router",
    "faqs_router",
    "health_router",
    "observability_router",
    "reports_router",
    "search_router",
    "tasks_router",
]
