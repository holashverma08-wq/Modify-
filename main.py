"""
PrepGalaxy — India's AI-Powered Telegram Government Exam Platform
====================================================================

PHASE STATUS: Phase 5 — AI Stack & Vision Engines

    This file currently implements:
        - Structured logging & Config mapping
        - Interfaces for Infrastructure (Cache, RateLimit, RBAC, AI, OCR, PDF)
        - ServiceContainer (Dependency Injection)
        - Concrete Infrastructure (DatabaseService, MemoryCacheManager, etc.)
        - Domain Services (`BaseService`, `UserService`, `AdminService`, `StudyService`)
        - ACID Transactions via MongoDB Sessions for referrals and admin operations.
        - Telegram Routers & Middlewares (Rate Limiting, Container Injection, Admin Filters)
        - Interactive User & Admin Dashboards
        - [PHASE 5]: Swappable AI Engine (Gemini), Vision OCR, OMR (OpenCV), PDF Engine.

ARCHITECTURE NOTES:
    - Infrastructure is completely decoupled behind Abstract Base Classes.
    - CPU-bound tasks (OpenCV, ReportLab) are securely dispatched via asyncio.to_thread().
    - AI interactions deduct internal currency via StudyService to gamify usage.
"""

from __future__ import annotations

import abc
import asyncio
import html
import io
import logging
import random
import string
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum
from logging.handlers import RotatingFileHandler
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    TypeVar,
)

import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command, BaseFilter
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ErrorEvent, InlineKeyboardButton, InlineKeyboardMarkup, BufferedInputFile
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorClientSession,
    AsyncIOMotorDatabase,
)
from pymongo import ASCENDING, IndexModel
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from config import (
    APP_NAME,
    APP_VERSION,
    LOG_BACKUP_COUNT,
    LOG_DIRECTORY,
    LOG_FILE_PATH,
    LOG_MAX_BYTES,
    Config,
    ConfigurationError,
    LogLevel,
)

# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

SYSTEM_HEALTH_COLLECTION: Final[str] = "system_health"
AUDIT_LOG_COLLECTION: Final[str] = "audit_logs"
USERS_COLLECTION: Final[str] = "users"

DB_OPERATION_MAX_RETRY_ATTEMPTS: Final[int] = 3
DB_OPERATION_RETRY_BACKOFF_SECONDS: Final[float] = 0.5
DEFAULT_PAGE_SIZE: Final[int] = 20
MAX_PAGE_SIZE: Final[int] = 100

AI_COST_COINS: Final[int] = 5

class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    FAILED = "failed"

class Role(IntEnum):
    USER = 10
    MODERATOR = 20
    CONTENT_MANAGER = 30
    ADMIN = 40
    SUPER_ADMIN = 50

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PrepGalaxyError(Exception): pass
class ValidationError(PrepGalaxyError): pass
class PermissionDeniedError(PrepGalaxyError): pass
class RateLimitExceededError(PrepGalaxyError): pass
class EntityNotFoundError(PrepGalaxyError): pass
class InsufficientFundsError(PrepGalaxyError): pass

T = TypeVar("T")

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Validation & Sanitization
# ---------------------------------------------------------------------------

class Validator:
    @staticmethod
    def validate_telegram_id(user_id: Any) -> int:
        try:
            val = int(user_id)
            if val <= 0: raise ValueError
            return val
        except (ValueError, TypeError):
            raise ValidationError("Invalid Telegram ID format.")

class TextSanitizer:
    @staticmethod
    def escape_html(text: str) -> str:
        return html.escape(str(text)) if text else ""

# ---------------------------------------------------------------------------
# Core Interfaces (Abstract Contracts)
# ---------------------------------------------------------------------------

class AbstractCacheManager(abc.ABC):
    @abc.abstractmethod
    async def get(self, key: str) -> Optional[Any]: pass
    @abc.abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 60) -> None: pass
    @abc.abstractmethod
    async def delete(self, key: str) -> None: pass

class AbstractRateLimiter(abc.ABC):
    @abc.abstractmethod
    async def acquire(self, user_id: int, action_type: str = "msg") -> None: pass

class AbstractPermissionManager(abc.ABC):
    @abc.abstractmethod
    async def verify_permission(self, user_id: int, required_role: Role) -> None: pass

class AbstractAIEngine(abc.ABC):
    @abc.abstractmethod
    async def generate_explanation(self, text: str) -> str: pass

class AbstractOCREngine(abc.ABC):
    @abc.abstractmethod
    async def extract_text(self, image_bytes: bytes) -> str: pass

class AbstractOMREngine(abc.ABC):
    @abc.abstractmethod
    async def evaluate_sheet(self, image_bytes: bytes) -> Dict[str, Any]: pass

class AbstractPDFEngine(abc.ABC):
    @abc.abstractmethod
    async def generate_report(self, data: Dict[str, Any]) -> bytes: pass

# ---------------------------------------------------------------------------
# Database Layer
# ---------------------------------------------------------------------------

INDEX_REGISTRY: Final[Dict[str, List[IndexModel]]] = {
    SYSTEM_HEALTH_COLLECTION: [
        IndexModel([("timestamp", ASCENDING)], name="idx_timestamp_ttl", expireAfterSeconds=30 * 86_400),
    ],
    AUDIT_LOG_COLLECTION: [
        IndexModel([("admin_id", ASCENDING)], name="idx_admin_id"),
        IndexModel([("timestamp", ASCENDING)], name="idx_timestamp_audit_ttl", expireAfterSeconds=365 * 86_400)
    ],
    USERS_COLLECTION: [
        IndexModel([("telegram_id", ASCENDING)], unique=True, name="idx_telegram_id"),
        IndexModel([("referral_code", ASCENDING)], unique=True, sparse=True, name="idx_referral_code"),
        IndexModel([("exam_target", ASCENDING)], name="idx_exam_target"),
    ]
}

_RETRYABLE_DB_EXCEPTIONS: Final[tuple] = (AutoReconnect, ConnectionFailure, ServerSelectionTimeoutError)

class DatabaseService:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger = logging.getLogger("prepgalaxy.db")
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    @property
    def database(self) -> AsyncIOMotorDatabase:
        if self._db is None: raise RuntimeError("Database not connected.")
        return self._db

    async def connect(self) -> None:
        delay = self._config.mongo_retry_base_delay_seconds
        for attempt in range(1, self._config.mongo_max_retries + 1):
            try:
                self._client = AsyncIOMotorClient(
                    self._config.mongo_uri,
                    minPoolSize=self._config.mongo_min_pool_size,
                    maxPoolSize=self._config.mongo_max_pool_size,
                    serverSelectionTimeoutMS=self._config.mongo_server_selection_timeout_ms,
                )
                await self._client.admin.command("ping")
                self._db = self._client[self._config.mongo_db_name]
                self._logger.info("MongoDB connection established (attempt %d).", attempt)
                return
            except _RETRYABLE_DB_EXCEPTIONS:
                if attempt < self._config.mongo_max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        raise ConnectionError("Failed to connect to MongoDB.")

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._logger.info("MongoDB connection closed.")

    async def ensure_indexes(self) -> None:
        for collection_name, index_models in INDEX_REGISTRY.items():
            await self.database[collection_name].create_indexes(index_models)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncIOMotorClientSession]:
        if self._client is None: raise RuntimeError("Database not connected.")
        async with await self._client.start_session() as session:
            async with session.start_transaction():
                yield session

    async def _with_retry(self, op_name: str, coro: Callable[[], Awaitable[T]]) -> T:
        for attempt in range(1, 4):
            try: return await coro()
            except _RETRYABLE_DB_EXCEPTIONS:
                if attempt < 3: await asyncio.sleep(0.5 * attempt)
            except PyMongoError: raise
        raise ConnectionError(f"DB op {op_name} failed.")

    async def insert_one(self, col: str, doc: Dict[str, Any], session: Optional[AsyncIOMotorClientSession] = None) -> str:
        res = await self._with_retry("insert_one", lambda: self.database[col].insert_one(doc, session=session))
        return str(res.inserted_id)

    async def find_one(self, col: str, query: Dict[str, Any], session: Optional[AsyncIOMotorClientSession] = None) -> Optional[Dict[str, Any]]:
        return await self._with_retry("find_one", lambda: self.database[col].find_one(query, session=session))

    async def find_many(self, col: str, query: Dict[str, Any], limit: int = 0, session: Optional[AsyncIOMotorClientSession] = None) -> List[Dict[str, Any]]:
        return await self._with_retry("find_many", lambda: self.database[col].find(query, session=session).limit(limit).to_list(length=None))

    async def update_one(self, col: str, query: Dict[str, Any], update: Dict[str, Any], upsert: bool=False, session: Optional[AsyncIOMotorClientSession] = None) -> bool:
        res = await self._with_retry("update_one", lambda: self.database[col].update_one(query, update, upsert=upsert, session=session))
        return res.matched_count > 0 or res.upserted_id is not None

    async def count_documents(self, col: str, query: Dict[str, Any], session: Optional[AsyncIOMotorClientSession] = None) -> int:
        return await self._with_retry("count_documents", lambda: self.database[col].count_documents(query, session=session))

# ---------------------------------------------------------------------------
# Concrete Infrastructure Implementations
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    value: Any
    expires_at: float

class MemoryCacheManager(AbstractCacheManager):
    def __init__(self) -> None:
        self._store: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry and time.time() < entry.expires_at: return entry.value
            if entry: del self._store[key]
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 60) -> None:
        async with self._lock:
            self._store[key] = CacheEntry(value, time.time() + ttl_seconds)

    async def delete(self, key: str) -> None:
        async with self._lock: self._store.pop(key, None)

    async def start_cleanup(self) -> None:
        async def _loop():
            while True:
                await asyncio.sleep(60)
                now = time.time()
                async with self._lock:
                    keys = [k for k, v in self._store.items() if now > v.expires_at]
                    for k in keys: del self._store[k]
        self._task = asyncio.create_task(_loop())

    def stop_cleanup(self) -> None:
        if self._task: self._task.cancel()

class SlidingWindowRateLimiter(AbstractRateLimiter):
    def __init__(self, cache: AbstractCacheManager, limit: int) -> None:
        self._cache = cache
        self._limit = limit

    async def acquire(self, user_id: int, action_type: str = "msg") -> None:
        key = f"rl:{action_type}:{user_id}"
        current = await self._cache.get(key) or 0
        if current >= self._limit: raise RateLimitExceededError("Spam protection triggered.")
        await self._cache.set(key, current + 1, ttl_seconds=1)

class DBBackedPermissionManager(AbstractPermissionManager):
    def __init__(self, db: DatabaseService, super_admins: List[int]) -> None:
        self._db = db
        self._super_admins = super_admins

    async def verify_permission(self, user_id: int, required_role: Role) -> None:
        if user_id in self._super_admins: return
        user = await self._db.find_one(USERS_COLLECTION, {"telegram_id": user_id})
        if not user: raise EntityNotFoundError("User not found.")
        if user.get("role", Role.USER.value) < required_role.value:
            raise PermissionDeniedError("Insufficient permissions.")

class AuditService:
    def __init__(self, db: DatabaseService) -> None:
        self._db = db

    async def log_action(self, admin_id: int, action: str, details: Dict[str, Any], session: Optional[AsyncIOMotorClientSession] = None) -> None:
        await self._db.insert_one(AUDIT_LOG_COLLECTION, {
            "admin_id": admin_id, "action": action, "details": details, "timestamp": utcnow()
        }, session=session)

# --- Phase 5: AI & Vision Infrastructure ---

class GeminiAIEngine(AbstractAIEngine):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    async def generate_explanation(self, text: str) -> str:
        prompt = f"Act as an expert instructor for Indian Government Exams (like UPSC, SSC CGL). Briefly explain: {text}"
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            raise PrepGalaxyError(f"AI Generation failed: {e}")

class GeminiOCREngine(AbstractOCREngine):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    async def extract_text(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            prompt = "Extract all text from this image precisely. Do not add conversational text."
            response = await self.model.generate_content_async([prompt, image])
            return response.text
        except Exception as e:
            raise PrepGalaxyError(f"OCR Extraction failed: {e}")

class OpenCVOMREngine(AbstractOMREngine):
    def _process_sync(self, image_bytes: bytes) -> Dict[str, Any]:
        """CPU-bound task wrapped for threadpool execution."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise PrepGalaxyError("Invalid image data provided for OMR.")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        
        # Identify contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles_found = 0
        
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
                bubbles_found += 1
                
        # Heuristic scoring simulation for UPSC / RAS structure defaults
        return {
            "total_marked_bubbles": bubbles_found,
            "estimated_score": bubbles_found * 2.0, # UPSC standard positive mark
            "confidence": "HIGH" if bubbles_found > 0 else "LOW"
        }

    async def evaluate_sheet(self, image_bytes: bytes) -> Dict[str, Any]:
        return await asyncio.to_thread(self._process_sync, image_bytes)

class ReportLabPDFEngine(AbstractPDFEngine):
    def _generate_sync(self, data: Dict[str, Any]) -> bytes:
        """CPU-bound task generating a binary PDF."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph(f"<b>{APP_NAME} - Performance Analytics</b>", styles['Title']))
        elements.append(Spacer(1, 12))
        
        student_info = f"<b>Student ID:</b> {data.get('telegram_id', 'Unknown')}<br/>" \
                       f"<b>Exam Target:</b> {data.get('exam_target', 'UPSC CDS / RAS')}<br/>" \
                       f"<b>Date:</b> {utcnow().strftime('%Y-%m-%d')}"
        elements.append(Paragraph(student_info, styles['Normal']))
        elements.append(Spacer(1, 24))
        
        table_data = [['Metric', 'Value']]
        for key, value in data.get('metrics', {}).items():
            table_data.append([str(key).replace('_', ' ').title(), str(value)])
            
        t = Table(table_data, colWidths=[200, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        
        doc.build(elements)
        return buffer.getvalue()

    async def generate_report(self, data: Dict[str, Any]) -> bytes:
        return await asyncio.to_thread(self._generate_sync, data)

# ---------------------------------------------------------------------------
# Dependency Injection Container
# ---------------------------------------------------------------------------

@dataclass
class ServiceContainer:
    config: Config
    db: DatabaseService
    cache: AbstractCacheManager
    rate_limiter: AbstractRateLimiter
    permissions: AbstractPermissionManager
    audit: AuditService
    ai_engine: AbstractAIEngine
    ocr_engine: AbstractOCREngine
    omr_engine: AbstractOMREngine
    pdf_engine: AbstractPDFEngine
    bot: Bot

class BaseService(abc.ABC):
    def __init__(self, container: ServiceContainer) -> None:
        self.container = container
        self.db = container.db
        self.cache = container.cache
        self.logger = logging.getLogger(f"prepgalaxy.service.{self.__class__.__name__.lower()}")

# ---------------------------------------------------------------------------
# Domain Services
# ---------------------------------------------------------------------------

class UserService(BaseService):
    async def _generate_collision_safe_referral(self) -> str:
        for _ in range(5):
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            if not await self.db.find_one(USERS_COLLECTION, {"referral_code": code}):
                return code
        raise PrepGalaxyError("Failed to generate unique referral code.")

    async def register_or_update(self, tg_user: types.User, ref_code: Optional[str] = None) -> Dict[str, Any]:
        user_id = Validator.validate_telegram_id(tg_user.id)
        existing_user = await self.db.find_one(USERS_COLLECTION, {"telegram_id": user_id})
        
        if existing_user:
            await self.db.update_one(USERS_COLLECTION, {"telegram_id": user_id}, {"$set": {"last_active": utcnow()}})
            return existing_user

        referral_code = await self._generate_collision_safe_referral()
        
        new_doc = {
            "telegram_id": user_id,
            "username": tg_user.username,
            "full_name": tg_user.full_name,
            "language": tg_user.language_code,
            "join_date": utcnow(),
            "last_active": utcnow(),
            "role": Role.USER.value,
            "coins": 200,
            "xp": 0,
            "level": 1,
            "exam_target": None,
            "premium_status": False,
            "referral_code": referral_code,
            "referred_by": None
        }

        async with self.db.transaction() as session:
            if ref_code:
                referrer = await self.db.find_one(USERS_COLLECTION, {"referral_code": ref_code}, session=session)
                if referrer and referrer["telegram_id"] != user_id:
                    new_doc["referred_by"] = referrer["telegram_id"]
                    await self.db.update_one(
                        USERS_COLLECTION, 
                        {"telegram_id": referrer["telegram_id"]}, 
                        {"$inc": {"coins": 50, "xp": 10}}, 
                        session=session
                    )
            await self.db.update_one(USERS_COLLECTION, {"telegram_id": user_id}, {"$set": new_doc}, upsert=True, session=session)
            
        return new_doc

    async def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        user = await self.db.find_one(USERS_COLLECTION, {"telegram_id": Validator.validate_telegram_id(user_id)})
        if not user: raise EntityNotFoundError("Profile not found.")
        return user

    async def set_exam_target(self, user_id: int, exam_name: str) -> None:
        await self.db.update_one(USERS_COLLECTION, {"telegram_id": user_id}, {"$set": {"exam_target": exam_name}})

class AdminService(BaseService):
    async def get_platform_stats(self) -> Dict[str, Any]:
        total_users = await self.db.count_documents(USERS_COLLECTION, {})
        premium_users = await self.db.count_documents(USERS_COLLECTION, {"premium_status": True})
        return {
            "total_users": total_users,
            "premium_users": premium_users,
            "timestamp": utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }

    async def modify_coins(self, admin_id: int, target_id: int, amount: int) -> None:
        target_id = Validator.validate_telegram_id(target_id)
        
        async with self.db.transaction() as session:
            success = await self.db.update_one(
                USERS_COLLECTION, 
                {"telegram_id": target_id}, 
                {"$inc": {"coins": amount}}, 
                session=session
            )
            if not success:
                raise EntityNotFoundError("Target user not found.")
                
            await self.container.audit.log_action(
                admin_id, "MODIFY_COINS", {"target": target_id, "amount": amount}, session=session
            )

    async def broadcast_message(self, admin_id: int, text: str) -> int:
        users = await self.db.find_many(USERS_COLLECTION, {})
        sent = 0
        for user in users:
            try:
                await self.container.bot.send_message(user["telegram_id"], text)
                sent += 1
                await asyncio.sleep(0.05) 
            except Exception as e:
                self.logger.warning("Broadcast failed for %s: %s", user["telegram_id"], e)
        
        await self.container.audit.log_action(
            admin_id, "BROADCAST", {"text_preview": text[:50], "delivered": sent}
        )
        return sent

class StudyService(BaseService):
    """Handles AI inference and gamification usage rules."""
    
    async def ask_ai_doubt(self, user_id: int, question: str) -> str:
        user_id = Validator.validate_telegram_id(user_id)
        user = await self.db.find_one(USERS_COLLECTION, {"telegram_id": user_id})
        
        if not user: raise EntityNotFoundError("User not found.")
        if user.get("coins", 0) < AI_COST_COINS:
            raise InsufficientFundsError(f"You need at least {AI_COST_COINS} coins to use AI.")
            
        async with self.db.transaction() as session:
            await self.db.update_one(USERS_COLLECTION, {"telegram_id": user_id}, {"$inc": {"coins": -AI_COST_COINS}}, session=session)
            
        return await self.container.ai_engine.generate_explanation(question)

    async def extract_document_text(self, image_bytes: bytes) -> str:
        return await self.container.ocr_engine.extract_text(image_bytes)

    async def process_omr_sheet(self, user_id: int, image_bytes: bytes) -> bytes:
        """Processes OMR and returns a performance PDF report."""
        user = await self.db.find_one(USERS_COLLECTION, {"telegram_id": user_id})
        target = user.get("exam_target", "UPSC CDS") if user else "Unknown"
        
        evaluation = await self.container.omr_engine.evaluate_sheet(image_bytes)
        
        pdf_data = {
            "telegram_id": user_id,
            "exam_target": target,
            "metrics": {
                "Total Marked Bubbles": evaluation["total_marked_bubbles"],
                "Estimated Score": f"+{evaluation['estimated_score']} Points",
                "Confidence": evaluation["confidence"]
            }
        }
        
        return await self.container.pdf_engine.generate_report(pdf_data)

# ---------------------------------------------------------------------------
# Telegram UI & Callback Data
# ---------------------------------------------------------------------------

class ExamCB(CallbackData, prefix="exm"): target: str
class NavCB(CallbackData, prefix="nav"): to: str

def build_exam_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎯 UPSC CDS", callback_data=ExamCB(target="UPSC CDS").pack()),
         InlineKeyboardButton(text="🎯 SSC CGL", callback_data=ExamCB(target="SSC CGL").pack())],
        [InlineKeyboardButton(text="🎯 RAS", callback_data=ExamCB(target="RAS").pack()),
         InlineKeyboardButton(text="🎯 Rajasthan LDC", callback_data=ExamCB(target="Rajasthan LDC").pack())],
        [InlineKeyboardButton(text="Other / Later", callback_data=NavCB(to="dash").pack())]
    ])

def build_dashboard_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👤 My Profile", callback_data=NavCB(to="profile").pack()),
         InlineKeyboardButton(text="⚙️ Settings", callback_data=NavCB(to="settings").pack())],
        [InlineKeyboardButton(text="📚 Notes & Materials", callback_data="feature_locked"),
         InlineKeyboardButton(text="📝 Mock Tests", callback_data="feature_locked")]
    ])

# ---------------------------------------------------------------------------
# Telegram Middlewares & Filters
# ---------------------------------------------------------------------------

class ContainerMiddleware(BaseMiddleware):
    def __init__(self, container: ServiceContainer):
        self.container = container

    async def __call__(self, handler, event, data):
        data["container"] = self.container
        data["user_service"] = UserService(self.container)
        data["admin_service"] = AdminService(self.container)
        data["study_service"] = StudyService(self.container)
        return await handler(event, data)

class RateLimitMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        container: ServiceContainer = data["container"]
        user_id = event.from_user.id if event.from_user else None
        if user_id:
            try: await container.rate_limiter.acquire(user_id)
            except RateLimitExceededError:
                if isinstance(event, types.Message): await event.answer("⚠️ Action blocked: Please slow down.")
                elif isinstance(event, types.CallbackQuery): await event.answer("⚠️ Too fast!", show_alert=True)
                return
        return await handler(event, data)

class AdminFilter(BaseFilter):
    def __init__(self, required_role: Role = Role.ADMIN):
        self.required_role = required_role
        
    async def __call__(self, message: types.Message, container: ServiceContainer) -> bool:
        try:
            await container.permissions.verify_permission(message.from_user.id, self.required_role)
            return True
        except (PermissionDeniedError, EntityNotFoundError):
            return False

# ---------------------------------------------------------------------------
# Telegram Handlers
# ---------------------------------------------------------------------------

user_router = Router(name="user_system")
admin_router = Router(name="admin_system")
admin_router.message.filter(AdminFilter(Role.ADMIN))

# --- Admin Routes ---

@admin_router.message(Command("admin"))
async def cmd_admin(message: types.Message, admin_service: AdminService):
    stats = await admin_service.get_platform_stats()
    text = (
        f"🛡 <b>Enterprise Admin Console</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"👥 <b>Total Users:</b> {stats['total_users']}\n"
        f"💎 <b>Premium Users:</b> {stats['premium_users']}\n"
        f"⏱ <b>Last Updated:</b> {stats['timestamp']}\n\n"
        f"<b>Commands:</b>\n"
        f"/addcoins &lt;id&gt; &lt;amount&gt;\n"
        f"/broadcast &lt;message&gt;"
    )
    await message.answer(text)

@admin_router.message(Command("addcoins"))
async def cmd_addcoins(message: types.Message, admin_service: AdminService):
    args = message.text.split()
    if len(args) != 3:
        return await message.answer("Usage: <code>/addcoins &lt;telegram_id&gt; &lt;amount&gt;</code>")
    try:
        target_id = int(args[1])
        amount = int(args[2])
        await admin_service.modify_coins(message.from_user.id, target_id, amount)
        await message.answer(f"✅ Successfully added {amount} coins to user {target_id}.")
    except ValueError:
        await message.answer("⚠️ Invalid ID or amount.")
    except EntityNotFoundError:
        await message.answer("⚠️ User not found in database.")

@admin_router.message(Command("broadcast"))
async def cmd_broadcast(message: types.Message, admin_service: AdminService):
    text = message.text.replace("/broadcast ", "").strip()
    if not text or text == "/broadcast":
        return await message.answer("Usage: <code>/broadcast &lt;message text&gt;</code>")
    
    await message.answer("📡 Broadcast initiated. This may take time...")
    sent = await admin_service.broadcast_message(message.from_user.id, text)
    await message.answer(f"✅ Broadcast complete. Delivered to {sent} users.")

# --- User Routes (Core) ---

@user_router.message(CommandStart())
async def cmd_start(message: types.Message, user_service: UserService):
    args = message.text.split()[1] if len(message.text.split()) > 1 else None
    user_doc = await user_service.register_or_update(message.from_user, ref_code=args)
    name = TextSanitizer.escape_html(message.from_user.first_name)
    
    if not user_doc.get("exam_target"):
        msg = (f"🌟 Welcome to <b>PrepGalaxy</b>, {name}!\n\n"
               f"You have received a starter bonus of 🪙 <b>{user_doc['coins']} Coins</b>.\n"
               f"Select your target exam:")
        await message.answer(msg, reply_markup=build_exam_keyboard())
    else:
        msg = f"🌟 Welcome back, {name}!\nTarget: <b>{user_doc['exam_target']}</b>"
        await message.answer(msg, reply_markup=build_dashboard_keyboard())

@user_router.callback_query(ExamCB.filter())
async def handle_exam_selection(callback: types.CallbackQuery, callback_data: ExamCB, user_service: UserService):
    await user_service.set_exam_target(callback.from_user.id, callback_data.target)
    await callback.answer(f"Target set to {callback_data.target}!")
    await callback.message.edit_text(
        f"✅ Excellent! Your target is <b>{callback_data.target}</b>.\n\nExplore next:",
        reply_markup=build_dashboard_keyboard()
    )

@user_router.callback_query(NavCB.filter(F.to == "dash"))
async def show_dashboard(callback: types.CallbackQuery):
    await callback.message.edit_text("🎛 <b>PrepGalaxy Dashboard</b>", reply_markup=build_dashboard_keyboard())
    await callback.answer()

@user_router.callback_query(NavCB.filter(F.to == "profile"))
async def show_profile(callback: types.CallbackQuery, user_service: UserService):
    prof = await user_service.get_user_profile(callback.from_user.id)
    text = (
        f"👤 <b>Your Profile</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Target:</b> {prof.get('exam_target', 'Not Set')}\n"
        f"⭐ <b>Level:</b> {prof.get('level', 1)} | <b>XP:</b> {prof.get('xp', 0)}\n"
        f"🪙 <b>Coins:</b> {prof.get('coins', 0)}\n"
        f"💎 <b>Premium:</b> {'Active' if prof.get('premium_status') else 'Free'}\n\n"
        f"🔗 <b>Invite Link:</b>\n"
        f"<code>https://t.me/{(await callback.bot.get_me()).username}?start={prof.get('referral_code')}</code>"
    )
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="⬅️ Back to Dashboard", callback_data=NavCB(to="dash").pack())
    ]])
    await callback.message.edit_text(text, reply_markup=kb)
    await callback.answer()

@user_router.callback_query(F.data == "feature_locked")
async def handle_locked(callback: types.CallbackQuery):
    await callback.answer("🔒 This feature is unlocking soon in the next update!", show_alert=True)

# --- Phase 5: AI Routes ---

@user_router.message(Command("ask"))
async def cmd_ask(message: types.Message, study_service: StudyService):
    question = message.text.replace("/ask ", "").strip()
    if not question or question == "/ask":
        return await message.answer("Usage: <code>/ask &lt;your doubt here&gt;</code>")
        
    wait_msg = await message.answer("🧠 AI is analyzing your question... (Cost: 5 Coins)")
    try:
        answer = await study_service.ask_ai_doubt(message.from_user.id, question)
        await wait_msg.edit_text(f"💡 <b>Explanation:</b>\n\n{answer}")
    except InsufficientFundsError as e:
        await wait_msg.edit_text(f"❌ {str(e)}")
    except Exception as e:
        await wait_msg.edit_text("⚠️ An error occurred while contacting the AI.")

@user_router.message(Command("omr"))
async def cmd_omr_request(message: types.Message):
    await message.answer("📸 Please send a photo of your filled OMR sheet and reply to it with `/scan_omr`")

@user_router.message(Command("scan_omr"))
async def cmd_scan_omr(message: types.Message, study_service: StudyService, bot: Bot):
    if not message.reply_to_message or not message.reply_to_message.photo:
        return await message.answer("⚠️ You must reply to an image of an OMR sheet with this command.")
        
    wait_msg = await message.answer("🔍 Scanning OMR and generating analytics PDF...")
    
    # Download photo to memory
    photo = message.reply_to_message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    image_bytes = downloaded_file.read()
    
    try:
        pdf_bytes = await study_service.process_omr_sheet(message.from_user.id, image_bytes)
        pdf_file = BufferedInputFile(pdf_bytes, filename=f"OMR_Result_{utcnow().strftime('%Y%m%d')}.pdf")
        
        await message.answer_document(pdf_file, caption="✅ Your OMR Analytics Report.")
        await wait_msg.delete()
    except Exception as e:
        await wait_msg.edit_text(f"⚠️ OMR Processing failed: {str(e)}")

# ---------------------------------------------------------------------------
# Application Initialization & Setup
# ---------------------------------------------------------------------------

def setup_logging(config: Config) -> None:
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(config.log_level.to_logging_level())
    root.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if config.log_level is not LogLevel.DEBUG:
        logging.getLogger("aiogram").setLevel(logging.WARNING)

class PrepGalaxyApplication:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger = logging.getLogger("prepgalaxy.app")

        self._bot = Bot(token=config.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self._dispatcher = Dispatcher(storage=MemoryStorage())
        self._dispatcher.errors.register(self._handle_global_error)

        db = DatabaseService(config)
        cache = MemoryCacheManager()
        rl = SlidingWindowRateLimiter(cache, config.rate_limit_mps)
        rbac = DBBackedPermissionManager(db, config.admin_ids)
        audit = AuditService(db)
        
        # Phase 5: Initialize AI & Vision Stack
        ai_engine = GeminiAIEngine(config.gemini_api_key)
        ocr_engine = GeminiOCREngine(config.gemini_api_key)
        omr_engine = OpenCVOMREngine()
        pdf_engine = ReportLabPDFEngine()

        self.container = ServiceContainer(
            config, db, cache, rl, rbac, audit, 
            ai_engine, ocr_engine, omr_engine, pdf_engine, 
            self._bot
        )

        self._dispatcher.update.outer_middleware(ContainerMiddleware(self.container))
        self._dispatcher.update.outer_middleware(RateLimitMiddleware())
        
        self._dispatcher.include_router(admin_router)
        self._dispatcher.include_router(user_router)

    async def _handle_global_error(self, event: ErrorEvent) -> bool:
        if isinstance(event.exception, PrepGalaxyError): return True
        self._logger.error("Unhandled exception: %s", event.exception, exc_info=True)
        return True

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try: loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._shutdown_signal(s)))
            except NotImplementedError: pass

    async def _shutdown_signal(self, sig) -> None:
        self._logger.warning("Shutdown signal %s received.", sig.name)
        self._dispatcher.stop_polling()

    async def startup(self) -> None:
        self._logger.info("Initializing PrepGalaxy...")
        await self.container.db.connect()
        await self.container.db.ensure_indexes()
        await self.container.cache.start_cleanup()
        self._logger.info("Bot Online: @%s", (await self._bot.get_me()).username)

    async def shutdown(self) -> None:
        self._logger.info("Shutting down safely...")
        self.container.cache.stop_cleanup()
        await self.container.db.disconnect()
        await self._bot.session.close()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._install_signal_handlers(loop)
        await self.startup()
        try:
            await self._dispatcher.start_polling(self._bot, handle_signals=False, close_bot_session=False)
        finally:
            await self.shutdown()

def main() -> None:
    try: config = Config.load()
    except ConfigurationError as exc:
        sys.exit(f"Fatal Config Error: {exc}")
    
    setup_logging(config)
    app = PrepGalaxyApplication(config)
    try: asyncio.run(app.run())
    except KeyboardInterrupt: pass
    except Exception as exc: logging.getLogger().critical("Fatal: %s", exc, exc_info=True)

if __name__ == "__main__":
    main()
