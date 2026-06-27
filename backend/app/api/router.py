from fastapi import APIRouter

from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.departments import router as departments_router
from app.api.routes.documents import router as documents_router
from app.api.routes.eval import router as eval_router
from app.api.routes.faqs import router as faqs_router
from app.api.routes.health import router as health_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.observability import router as observability_router
from app.api.routes.permissions import router as permissions_router
from app.api.routes.reports import router as reports_router
from app.api.routes.search import router as search_router
from app.api.routes.tasks import router as tasks_router
from app.api.routes.users import router as users_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(documents_router)
api_router.include_router(departments_router)
api_router.include_router(users_router)
api_router.include_router(search_router)
api_router.include_router(chat_router)
api_router.include_router(tasks_router)
api_router.include_router(reports_router)
api_router.include_router(faqs_router)
api_router.include_router(eval_router)
api_router.include_router(jobs_router)
api_router.include_router(observability_router)
api_router.include_router(permissions_router)
