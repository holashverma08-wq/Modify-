"""
PrepGalaxy — India's Telegram Government Exam Platform
========================================================
PHASE 1 / N — Core Foundation

This file is intentionally the ONLY Python module in the project (plus
requirements.txt), per project constraints. Everything — config, database,
services, handlers — lives here in clearly labeled sections.

PHASE 1 SCOPE (this file, working end-to-end today):
    - Configuration (.env driven)
    - Structured logging
    - MongoDB (Motor) data layer + indexes
    - Rate limiting / anti-spam middleware
    - Audit logging
    - Global error handling
    - Smart onboarding (FSM)
    - Student profile
    - Referral system (core mechanics)
    - Leaderboard (basic)
    - Admin: /stats, /broadcast, /ban, /unban

NOT YET IN THIS FILE (planned follow-up phases — see chat for the plan):
    - AI Doubt Solver (Gemini) + Gemini Vision
    - OCR / OMR checking pipeline (OpenCV, Tesseract)
    - Mock Test Engine + AI Analytics
    - Premium PDF Reports (ReportLab, PyMuPDF, Matplotlib)
    - XP/Coins economy tuning, Gamification badges
    - Premium membership / payments hooks
    - Notes management
    - Result search engine / PDF result parsing

Required environment variables (.env):
    BOT_TOKEN               Telegram bot token from @BotFather
    MONGODB_URI              MongoDB Atlas connection string
    MONGODB_DB_NAME           Database name (default: prepgalaxy)
    OWNER_TELEGRAM_ID        Telegram user id of the platform owner (int)
    ADMIN_TELEGRAM_IDS       Comma-separated list of admin telegram ids
    GEMINI_API_KEY           (loaded now, used starting Phase 2)
    LOG_LEVEL                 DEBUG|INFO|WARNING|ERROR (default: INFO)
    ENVIRONMENT               development|production (default: development)
"""

from __future__ import annotations

# =====================================================================
# SECTION: IMPORTS
# =====================================================================
import asyncio
import logging
import os
import secrets
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError

from aiogram import Bot, Dispatcher, F, Router, BaseMiddleware
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardMarkup,
    Message,
    TelegramObject,
    Update,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder


# =====================================================================
# SECTION: CONFIGURATION
# =====================================================================
@dataclass(frozen=True)
class Settings:
    """Strongly-typed application configuration loaded from environment."""

    bot_token: str
    mongodb_uri: str
    mongodb_db_name: str
    owner_telegram_id: int
    admin_telegram_ids: List[int]
    gemini_api_key: Optional[str]
    log_level: str
    environment: str

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


def _parse_admin_ids(raw: str) -> List[int]:
    """Parse a comma-separated string of Telegram admin IDs into ints."""
    ids: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.append(int(chunk))
        except ValueError:
            logging.warning("Ignoring invalid admin id in ADMIN_TELEGRAM_IDS: %r", chunk)
    return ids


def load_settings() -> Settings:
    """Load and validate configuration from the environment (.env)."""
    load_dotenv()

    bot_token = os.getenv("BOT_TOKEN", "").strip()
    mongodb_uri = os.getenv("MONGODB_URI", "").strip()
    mongodb_db_name = os.getenv("MONGODB_DB_NAME", "prepgalaxy").strip()
    owner_raw = os.getenv("OWNER_TELEGRAM_ID", "0").strip()
    admin_raw = os.getenv("ADMIN_TELEGRAM_IDS", "")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    environment = os.getenv("ENVIRONMENT", "development").strip()

    missing = [
        name
        for name, value in (("BOT_TOKEN", bot_token), ("MONGODB_URI", mongodb_uri))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Create a .env file (see module docstring for the full list)."
        )

    try:
        owner_telegram_id = int(owner_raw)
    except ValueError:
        raise RuntimeError("OWNER_TELEGRAM_ID must be an integer.") from None

    admin_ids = _parse_admin_ids(admin_raw)
    if owner_telegram_id and owner_telegram_id not in admin_ids:
        admin_ids.append(owner_telegram_id)

    return Settings(
        bot_token=bot_token,
        mongodb_uri=mongodb_uri,
        mongodb_db_name=mongodb_db_name,
        owner_telegram_id=owner_telegram_id,
        admin_telegram_ids=admin_ids,
        gemini_api_key=gemini_api_key,
        log_level=log_level,
        environment=environment,
    )


# =====================================================================
# SECTION: LOGGING
# =====================================================================
def configure_logging(settings: Settings) -> logging.Logger:
    """Configure structured logging to console + rotating file."""
    logger = logging.getLogger("prepgalaxy")
    logger.setLevel(settings.log_level)
    logger.propagate = False

    if logger.handlers:
        return logger  # Already configured (e.g., re-imported in tests).

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    try:
        os.makedirs("logs", exist_ok=True)
        file_handler = RotatingFileHandler(
            "logs/prepgalaxy.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not attach file log handler: %s", exc)

    return logger


logger = logging.getLogger("prepgalaxy")


# =====================================================================
# SECTION: CONSTANTS / DOMAIN ENUMS
# =====================================================================
TARGET_EXAMS: List[str] = [
    "LDC", "RAS", "SSC", "REET", "Police",
    "Patwari", "CET", "Railway", "Bank", "Teaching",
]

LANGUAGES: List[str] = ["English", "Hindi", "Hinglish"]

SIGNUP_XP_BONUS = 50
SIGNUP_COIN_BONUS = 20
REFERRAL_XP_BONUS = 100
REFERRAL_COIN_BONUS = 50


class AuditAction(str, Enum):
    USER_ONBOARDED = "user_onboarded"
    REFERRAL_APPLIED = "referral_applied"
    ADMIN_BROADCAST = "admin_broadcast"
    ADMIN_BAN = "admin_ban"
    ADMIN_UNBAN = "admin_unban"
    RATE_LIMITED = "rate_limited"
    UNHANDLED_ERROR = "unhandled_error"


def calculate_level(xp: int) -> int:
    """Simple level curve: level up every 200 XP, minimum level 1."""
    return max(1, (xp // 200) + 1)


def generate_referral_code() -> str:
    """Generate a short, human-shareable referral code."""
    return f"PG{secrets.token_hex(3).upper()}"


# =====================================================================
# SECTION: DATABASE
# =====================================================================
class Database:
    """Owns the Motor client and exposes typed collection handles."""

    def __init__(self, settings: Settings) -> None:
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(
            settings.mongodb_uri,
            maxPoolSize=50,
            minPoolSize=5,
            serverSelectionTimeoutMS=8000,
        )
        self.db: AsyncIOMotorDatabase = self._client[settings.mongodb_db_name]

        self.users: AsyncIOMotorCollection = self.db["users"]
        self.audit_logs: AsyncIOMotorCollection = self.db["audit_logs"]
        self.broadcasts: AsyncIOMotorCollection = self.db["broadcasts"]

    async def ping(self) -> None:
        await self._client.admin.command("ping")

    async def create_indexes(self) -> None:
        """Create/ensure all indexes. Idempotent — safe to call on every boot."""
        await self.users.create_index("telegram_id", unique=True, name="uniq_telegram_id")
        await self.users.create_index("referral_code", unique=True, name="uniq_referral_code")
        await self.users.create_index([("xp", DESCENDING)], name="xp_leaderboard")
        await self.users.create_index([("target_exam", ASCENDING)], name="target_exam_idx")
        await self.audit_logs.create_index(
            "created_at", expireAfterSeconds=60 * 60 * 24 * 90, name="ttl_90d"
        )
        await self.audit_logs.create_index([("telegram_id", ASCENDING)], name="audit_user_idx")
        logger.info("MongoDB indexes ensured.")

    async def close(self) -> None:
        self._client.close()
        logger.info("MongoDB connection closed.")


class UserRepository:
    """Data-access layer for the `users` collection."""

    def __init__(self, database: Database) -> None:
        self._col = database.users

    async def get_by_telegram_id(self, telegram_id: int) -> Optional[Dict[str, Any]]:
        return await self._col.find_one({"telegram_id": telegram_id})

    async def get_by_referral_code(self, code: str) -> Optional[Dict[str, Any]]:
        return await self._col.find_one({"referral_code": code})

    async def create_user(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await self._col.insert_one(doc)
        except DuplicateKeyError:
            logger.warning("Duplicate user insert attempted for telegram_id=%s", doc.get("telegram_id"))
        return doc

    async def touch_last_active(self, telegram_id: int) -> None:
        await self._col.update_one(
            {"telegram_id": telegram_id},
            {"$set": {"last_active_at": datetime.now(timezone.utc)}},
        )

    async def add_xp_and_coins(self, telegram_id: int, xp: int, coins: int) -> None:
        await self._col.update_one(
            {"telegram_id": telegram_id},
            {"$inc": {"xp": xp, "coins": coins}},
        )

    async def set_banned(self, telegram_id: int, banned: bool) -> bool:
        result = await self._col.update_one(
            {"telegram_id": telegram_id}, {"$set": {"is_banned": banned}}
        )
        return result.modified_count > 0

    async def count_all(self) -> int:
        return await self._col.count_documents({})

    async def count_active_since(self, since: datetime) -> int:
        return await self._col.count_documents({"last_active_at": {"$gte": since}})

    async def top_by_xp(self, limit: int = 10) -> List[Dict[str, Any]]:
        cursor = self._col.find({}).sort("xp", DESCENDING).limit(limit)
        return [doc async for doc in cursor]

    def iter_broadcast_targets(self):
        """Async cursor over non-banned users, for broadcast fan-out."""
        return self._col.find({"is_banned": {"$ne": True}}, {"telegram_id": 1})


class AuditLogger:
    """Append-only audit trail with automatic 90-day TTL expiry."""

    def __init__(self, database: Database) -> None:
        self._col = database.audit_logs

    async def log(
        self, telegram_id: Optional[int], action: AuditAction, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        try:
            await self._col.insert_one(
                {
                    "telegram_id": telegram_id,
                    "action": action.value,
                    "metadata": metadata or {},
                    "created_at": datetime.now(timezone.utc),
                }
            )
        except PyMongoError as exc:
            logger.error("Failed to write audit log (%s): %s", action.value, exc)


# =====================================================================
# SECTION: UTILITIES (rate limiting / anti-spam / validation)
# =====================================================================
class RateLimiter:
    """
    In-process sliding-window rate limiter.

    NOTE: This is process-local (in-memory), which is correct for a single
    polling worker. If PrepGalaxy is horizontally scaled across multiple
    processes/machines in a later phase, this should be swapped for a
    shared store — the locked tech stack for this project does not include
    Redis, so a Mongo-backed limiter is the documented upgrade path.
    """

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def is_allowed(self, key: str, limit: int, window_seconds: float) -> bool:
        now = time.monotonic()
        bucket = self._hits[key]
        while bucket and now - bucket[0] > window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def validate_full_name(text: str) -> Optional[str]:
    """Return a cleaned name, or None if invalid."""
    cleaned = " ".join(text.strip().split())
    if 2 <= len(cleaned) <= 60 and all(c.isalpha() or c.isspace() or c in ".-'" for c in cleaned):
        return cleaned
    return None


def validate_region(text: str) -> Optional[str]:
    cleaned = " ".join(text.strip().split())
    if 2 <= len(cleaned) <= 60:
        return cleaned
    return None


# =====================================================================
# SECTION: KEYBOARDS
# =====================================================================
def build_exam_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for exam in TARGET_EXAMS:
        builder.button(text=exam, callback_data=f"exam:{exam}")
    builder.adjust(2)
    return builder.as_markup()


def build_language_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for lang in LANGUAGES:
        builder.button(text=lang, callback_data=f"lang:{lang}")
    builder.adjust(3)
    return builder.as_markup()


def build_main_menu_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="👤 My Profile", callback_data="menu:profile")
    builder.button(text="🏆 Leaderboard", callback_data="menu:leaderboard")
    builder.button(text="🎁 Referrals", callback_data="menu:referral")
    builder.adjust(1)
    return builder.as_markup()


# =====================================================================
# SECTION: FSM STATES
# =====================================================================
class OnboardingStates(StatesGroup):
    full_name = State()
    target_exam = State()
    region = State()
    language = State()


# =====================================================================
# SECTION: MIDDLEWARES
# =====================================================================
class ThrottlingMiddleware(BaseMiddleware):
    """Blocks a user who fires more than `limit` updates per `window` seconds."""

    def __init__(self, rate_limiter: RateLimiter, limit: int = 8, window_seconds: float = 10.0) -> None:
        self._rate_limiter = rate_limiter
        self._limit = limit
        self._window = window_seconds

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is not None:
            key = f"throttle:{user.id}"
            if not self._rate_limiter.is_allowed(key, self._limit, self._window):
                audit_logger: AuditLogger = data["audit_logger"]
                asyncio.create_task(
                    audit_logger.log(user.id, AuditAction.RATE_LIMITED, {"limit": self._limit})
                )
                if isinstance(event, Message):
                    await event.answer("⏳ You're going a bit fast — please wait a few seconds and try again.")
                elif isinstance(event, CallbackQuery):
                    await event.answer("⏳ Slow down a little!", show_alert=False)
                return None
        return await handler(event, data)


class ErrorHandlingMiddleware(BaseMiddleware):
    """Catches unhandled exceptions from any handler so the bot never crashes silently."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        try:
            return await handler(event, data)
        except Exception as exc:  # noqa: BLE001 — top-level safety net, by design
            logger.exception("Unhandled error while processing update: %s", exc)
            audit_logger: AuditLogger = data.get("audit_logger")
            user = data.get("event_from_user")
            if audit_logger is not None:
                await audit_logger.log(
                    getattr(user, "id", None),
                    AuditAction.UNHANDLED_ERROR,
                    {"error": str(exc), "type": type(exc).__name__},
                )
            target = None
            if isinstance(event, Message):
                target = event
            elif isinstance(event, CallbackQuery):
                target = event.message
            if target is not None:
                try:
                    await target.answer(
                        "⚠️ Something went wrong on our end. Our team has been notified — please try again."
                    )
                except Exception:  # noqa: BLE001 — best-effort user notification
                    pass
            return None


# =====================================================================
# SECTION: USER HANDLERS
# =====================================================================
user_router = Router(name="user")


def _is_admin(settings: Settings, telegram_id: int) -> bool:
    return telegram_id in settings.admin_telegram_ids


@user_router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, user_repo: UserRepository, audit_logger: AuditLogger) -> None:
    """Entry point. Handles fresh signups (with optional referral payload) and returning users."""
    telegram_id = message.from_user.id
    existing = await user_repo.get_by_telegram_id(telegram_id)

    if existing:
        await user_repo.touch_last_active(telegram_id)
        await message.answer(
            f"👋 Welcome back, <b>{existing.get('full_name', 'Student')}</b>!\n"
            f"🎯 Target: {existing.get('target_exam', '—')}\n"
            f"✨ XP: {existing.get('xp', 0)} | 🪙 Coins: {existing.get('coins', 0)}",
            reply_markup=build_main_menu_keyboard(),
        )
        return

    referral_code: Optional[str] = None
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) > 1 and parts[1].startswith("ref_"):
        referral_code = parts[1][4:].strip().upper()

    await state.update_data(referral_code=referral_code)
    await state.set_state(OnboardingStates.full_name)
    await message.answer(
        "🎓 <b>Welcome to PrepGalaxy!</b>\n\n"
        "India's Telegram platform for LDC, RAS, SSC, REET, Police, Patwari, CET, "
        "Railway, Bank & Teaching exam prep.\n\n"
        "Let's set up your profile. What's your <b>full name</b>?"
    )


@user_router.message(StateFilter(OnboardingStates.full_name))
async def onboarding_full_name(message: Message, state: FSMContext) -> None:
    name = validate_full_name(message.text or "")
    if not name:
        await message.answer("That doesn't look like a valid name. Please enter your full name (letters only).")
        return
    await state.update_data(full_name=name)
    await state.set_state(OnboardingStates.target_exam)
    await message.answer(f"Nice to meet you, {name}! 🎯 Which exam are you preparing for?", reply_markup=build_exam_keyboard())


@user_router.callback_query(StateFilter(OnboardingStates.target_exam), F.data.startswith("exam:"))
async def onboarding_target_exam(callback: CallbackQuery, state: FSMContext) -> None:
    exam = callback.data.split(":", maxsplit=1)[1]
    await state.update_data(target_exam=exam)
    await state.set_state(OnboardingStates.region)
    await callback.message.edit_text(f"Target set: <b>{exam}</b> ✅\n\nWhich state/region are you preparing from?")
    await callback.answer()


@user_router.message(StateFilter(OnboardingStates.region))
async def onboarding_region(message: Message, state: FSMContext) -> None:
    region = validate_region(message.text or "")
    if not region:
        await message.answer("Please enter a valid state/region name.")
        return
    await state.update_data(region=region)
    await state.set_state(OnboardingStates.language)
    await message.answer("Preferred language for content?", reply_markup=build_language_keyboard())


@user_router.callback_query(StateFilter(OnboardingStates.language), F.data.startswith("lang:"))
async def onboarding_language(
    callback: CallbackQuery,
    state: FSMContext,
    user_repo: UserRepository,
    audit_logger: AuditLogger,
) -> None:
    language = callback.data.split(":", maxsplit=1)[1]
    data = await state.get_data()

    telegram_id = callback.from_user.id
    referral_code = data.get("referral_code")
    referrer: Optional[Dict[str, Any]] = None
    if referral_code:
        referrer = await user_repo.get_by_referral_code(referral_code)

    new_user_doc = {
        "telegram_id": telegram_id,
        "username": callback.from_user.username,
        "full_name": data["full_name"],
        "target_exam": data["target_exam"],
        "region": data["region"],
        "language": language,
        "xp": SIGNUP_XP_BONUS,
        "coins": SIGNUP_COIN_BONUS,
        "referral_code": generate_referral_code(),
        "referred_by": referrer["telegram_id"] if referrer else None,
        "is_admin": False,
        "is_banned": False,
        "created_at": datetime.now(timezone.utc),
        "last_active_at": datetime.now(timezone.utc),
    }
    await user_repo.create_user(new_user_doc)
    await audit_logger.log(telegram_id, AuditAction.USER_ONBOARDED, {"target_exam": data["target_exam"]})

    if referrer:
        await user_repo.add_xp_and_coins(referrer["telegram_id"], REFERRAL_XP_BONUS, REFERRAL_COIN_BONUS)
        await audit_logger.log(
            referrer["telegram_id"],
            AuditAction.REFERRAL_APPLIED,
            {"new_user_telegram_id": telegram_id},
        )

    await state.clear()
    await callback.message.edit_text(
        f"🎉 <b>You're all set, {new_user_doc['full_name']}!</b>\n\n"
        f"✨ +{SIGNUP_XP_BONUS} XP  |  🪙 +{SIGNUP_COIN_BONUS} Coins\n"
        f"🔗 Your referral code: <code>{new_user_doc['referral_code']}</code>\n\n"
        "Use the menu below to explore:",
    )
    await callback.message.answer("Main menu:", reply_markup=build_main_menu_keyboard())
    await callback.answer()


@user_router.message(Command("profile"))
async def cmd_profile(message: Message, user_repo: UserRepository) -> None:
    user = await user_repo.get_by_telegram_id(message.from_user.id)
    if not user:
        await message.answer("You haven't onboarded yet — send /start to begin.")
        return
    level = calculate_level(user.get("xp", 0))
    await message.answer(
        "👤 <b>Your Profile</b>\n\n"
        f"Name: {user.get('full_name')}\n"
        f"Target Exam: {user.get('target_exam')}\n"
        f"Region: {user.get('region')}\n"
        f"Language: {user.get('language')}\n"
        f"Level: {level}  |  ✨ XP: {user.get('xp', 0)}  |  🪙 Coins: {user.get('coins', 0)}\n"
        f"Referral Code: <code>{user.get('referral_code')}</code>"
    )


@user_router.message(Command("leaderboard"))
async def cmd_leaderboard(message: Message, user_repo: UserRepository) -> None:
    top_users = await user_repo.top_by_xp(limit=10)
    if not top_users:
        await message.answer("No students on the leaderboard yet — be the first!")
        return
    lines = ["🏆 <b>Top 10 Leaderboard</b>\n"]
    for rank, u in enumerate(top_users, start=1):
        lines.append(f"{rank}. {u.get('full_name', 'Student')} — {u.get('xp', 0)} XP")
    await message.answer("\n".join(lines))


@user_router.message(Command("referral"))
async def cmd_referral(message: Message, user_repo: UserRepository) -> None:
    user = await user_repo.get_by_telegram_id(message.from_user.id)
    if not user:
        await message.answer("You haven't onboarded yet — send /start to begin.")
        return
    bot_username = (await message.bot.get_me()).username
    link = f"https://t.me/{bot_username}?start=ref_{user['referral_code']}"
    await message.answer(
        "🎁 <b>Invite friends, earn rewards!</b>\n\n"
        f"You get +{REFERRAL_XP_BONUS} XP and +{REFERRAL_COIN_BONUS} coins per friend who joins.\n\n"
        f"Your link:\n{link}"
    )


@user_router.callback_query(F.data.startswith("menu:"))
async def on_menu_callback(callback: CallbackQuery, user_repo: UserRepository) -> None:
    action = callback.data.split(":", maxsplit=1)[1]
    if action == "profile":
        await cmd_profile(callback.message, user_repo)  # type: ignore[arg-type]
    elif action == "leaderboard":
        await cmd_leaderboard(callback.message, user_repo)  # type: ignore[arg-type]
    elif action == "referral":
        await cmd_referral(callback.message, user_repo)  # type: ignore[arg-type]
    await callback.answer()


@user_router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📚 <b>PrepGalaxy Commands</b>\n\n"
        "/start — Begin or return to the main menu\n"
        "/profile — View your profile & stats\n"
        "/leaderboard — Top 10 students by XP\n"
        "/referral — Get your referral link\n"
        "/help — Show this message"
    )


# =====================================================================
# SECTION: ADMIN COMMANDS
# =====================================================================
admin_router = Router(name="admin")


class BroadcastStates(StatesGroup):
    awaiting_content = State()


@admin_router.message(Command("stats"))
async def cmd_admin_stats(message: Message, settings: Settings, user_repo: UserRepository) -> None:
    if not _is_admin(settings, message.from_user.id):
        return
    total = await user_repo.count_all()
    active_24h = await user_repo.count_active_since(datetime.now(timezone.utc) - timedelta(hours=24))
    active_7d = await user_repo.count_active_since(datetime.now(timezone.utc) - timedelta(days=7))
    await message.answer(
        "📊 <b>Platform Stats</b>\n\n"
        f"Total students: {total}\n"
        f"Active (24h): {active_24h}\n"
        f"Active (7d): {active_7d}"
    )


@admin_router.message(Command("broadcast"))
async def cmd_broadcast_start(message: Message, state: FSMContext, settings: Settings) -> None:
    if not _is_admin(settings, message.from_user.id):
        return
    await state.set_state(BroadcastStates.awaiting_content)
    await message.answer("📢 Send me the message to broadcast to all students (text only, for now).")


@admin_router.message(StateFilter(BroadcastStates.awaiting_content))
async def cmd_broadcast_send(
    message: Message,
    state: FSMContext,
    settings: Settings,
    user_repo: UserRepository,
    audit_logger: AuditLogger,
) -> None:
    if not _is_admin(settings, message.from_user.id):
        await state.clear()
        return

    await state.clear()
    content = message.text or ""
    status_msg = await message.answer("🚀 Broadcasting… this may take a while for large audiences.")

    sent, failed = 0, 0
    async for user_doc in user_repo.iter_broadcast_targets():
        target_id = user_doc["telegram_id"]
        try:
            await message.bot.send_message(target_id, content)
            sent += 1
        except Exception as exc:  # noqa: BLE001 — per-recipient failure shouldn't stop the run
            failed += 1
            logger.debug("Broadcast failed for %s: %s", target_id, exc)
        await asyncio.sleep(0.05)  # ~20 msg/sec, safely under Telegram's limits

    await audit_logger.log(
        message.from_user.id, AuditAction.ADMIN_BROADCAST, {"sent": sent, "failed": failed}
    )
    await status_msg.edit_text(f"✅ Broadcast complete. Sent: {sent}  |  Failed: {failed}")


@admin_router.message(Command("ban"))
async def cmd_ban_user(message: Message, settings: Settings, user_repo: UserRepository, audit_logger: AuditLogger) -> None:
    if not _is_admin(settings, message.from_user.id):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2 or not parts[1].strip().lstrip("-").isdigit():
        await message.answer("Usage: /ban <telegram_id>")
        return
    target_id = int(parts[1].strip())
    ok = await user_repo.set_banned(target_id, True)
    if ok:
        await audit_logger.log(message.from_user.id, AuditAction.ADMIN_BAN, {"target": target_id})
        await message.answer(f"🚫 User {target_id} banned.")
    else:
        await message.answer("User not found.")


@admin_router.message(Command("unban"))
async def cmd_unban_user(message: Message, settings: Settings, user_repo: UserRepository, audit_logger: AuditLogger) -> None:
    if not _is_admin(settings, message.from_user.id):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2 or not parts[1].strip().lstrip("-").isdigit():
        await message.answer("Usage: /unban <telegram_id>")
        return
    target_id = int(parts[1].strip())
    ok = await user_repo.set_banned(target_id, False)
    if ok:
        await audit_logger.log(message.from_user.id, AuditAction.ADMIN_UNBAN, {"target": target_id})
        await message.answer(f"✅ User {target_id} unbanned.")
    else:
        await message.answer("User not found.")


# =====================================================================
# SECTION: BACKGROUND TASKS
# =====================================================================
async def periodic_health_log(database: Database, interval_seconds: int = 900) -> None:
    """Lightweight recurring heartbeat that also verifies DB connectivity stays healthy."""
    while True:
        try:
            await database.ping()
            logger.info("Heartbeat OK — MongoDB reachable.")
        except PyMongoError as exc:
            logger.error("Heartbeat FAILED — MongoDB unreachable: %s", exc)
        await asyncio.sleep(interval_seconds)


# =====================================================================
# SECTION: STARTUP
# =====================================================================
async def on_startup(bot: Bot, database: Database) -> None:
    await database.ping()
    await database.create_indexes()
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Begin / main menu"),
            BotCommand(command="profile", description="View your profile"),
            BotCommand(command="leaderboard", description="Top students by XP"),
            BotCommand(command="referral", description="Get your referral link"),
            BotCommand(command="help", description="List all commands"),
        ]
    )
    logger.info("PrepGalaxy startup complete.")


async def on_shutdown(database: Database) -> None:
    await database.close()
    logger.info("PrepGalaxy shutdown complete.")


# =====================================================================
# SECTION: MAIN FUNCTION
# =====================================================================
async def main() -> None:
    settings = load_settings()
    configure_logging(settings)
    logger.info("Starting PrepGalaxy in %s mode…", settings.environment)

    database = Database(settings)
    user_repo = UserRepository(database)
    audit_logger = AuditLogger(database)
    rate_limiter = RateLimiter()

    bot = Bot(token=settings.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dispatcher = Dispatcher(storage=MemoryStorage())

    # Dependency injection: available to every handler via function parameters.
    dispatcher["settings"] = settings
    dispatcher["database"] = database
    dispatcher["user_repo"] = user_repo
    dispatcher["audit_logger"] = audit_logger

    dispatcher.update.outer_middleware(ErrorHandlingMiddleware())
    dispatcher.update.outer_middleware(ThrottlingMiddleware(rate_limiter))

    dispatcher.include_router(admin_router)
    dispatcher.include_router(user_router)

    await on_startup(bot, database)
    health_task = asyncio.create_task(periodic_health_log(database))

    try:
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    finally:
        health_task.cancel()
        await on_shutdown(database)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("PrepGalaxy stopped by user.")
    except RuntimeError as exc:
        # Configuration errors (e.g., missing .env values) land here.
        print(f"Startup failed: {exc}", file=sys.stderr)
        sys.exit(1)
