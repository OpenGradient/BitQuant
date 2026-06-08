import asyncio
import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from supabase import AsyncClient, AsyncClientOptions, acreate_client

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_SECONDS = 10

# USD per token. This is used only to compute the existing aggregate cost_usd
MODEL_PRICES_USD: dict[str, tuple[Decimal, Decimal]] = {
    "google/gemini-2.5-flash": (Decimal("0.0000003"), Decimal("0.0000025")),
    "gemini-2.5-flash": (Decimal("0.0000003"), Decimal("0.0000025")),
    "gemini_2_5_flash": (Decimal("0.0000003"), Decimal("0.0000025")),
    "google/gemini-2.5-pro": (Decimal("0.00000125"), Decimal("0.00001")),
    "gemini-2.5-pro": (Decimal("0.00000125"), Decimal("0.00001")),
    "gemini_2_5_pro": (Decimal("0.00000125"), Decimal("0.00001")),
}


def _model_name(model: Any) -> str:
    value = getattr(model, "value", model)
    return str(value).lower()


def _decimal_or_zero(value: Decimal | str | float | None) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value if value.is_finite() and value >= 0 else Decimal("0")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() and parsed >= 0 else Decimal("0")


def _usage_metadata(message: Any) -> Optional[dict[str, Any]]:
    usage = getattr(message, "usage_metadata", None)
    return usage if isinstance(usage, dict) else None


def _tokens_from_usage(usage: dict[str, Any]) -> tuple[int, int]:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return input_tokens, output_tokens


def _estimate_cost_usd(
    *, model: Any, input_tokens: int, output_tokens: int
) -> Decimal:
    model_name = _model_name(model)
    prices = MODEL_PRICES_USD.get(model_name)
    if prices is None:
        logger.warning("No BitQuant LLM price configured for model %s", model_name)
        return Decimal("0")

    input_price, output_price = prices
    return Decimal(input_tokens) * input_price + Decimal(output_tokens) * output_price


def estimate_cost_from_message(*, model: Any, message: Any) -> Decimal:
    usage = _usage_metadata(message)
    if not usage:
        logger.warning("No LLM usage metadata found in BitQuant model response")
        return Decimal("0")

    input_tokens, output_tokens = _tokens_from_usage(usage)
    return _estimate_cost_usd(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def estimate_cost_from_messages(*, model: Any, messages: list[Any]) -> Decimal:
    input_tokens = 0
    output_tokens = 0
    saw_usage = False

    for message in messages:
        usage = _usage_metadata(message)
        if not usage:
            continue
        saw_usage = True
        message_input, message_output = _tokens_from_usage(usage)
        input_tokens += message_input
        output_tokens += message_output

    if not saw_usage:
        logger.warning("No LLM usage metadata found in BitQuant agent result messages")

    return _estimate_cost_usd(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class OhttpUsageBatcher:
    """Batch BitQuant usage into chat-api's existing OHTTP usage table.

    This writes the same aggregate shape as chat-api:
    ``request_count``, ``cost_usd``, and ``cost_opg``.
    """

    def __init__(self, *, supabase_url: str, service_role_key: str):
        self._supabase_url = supabase_url.rstrip("/")
        self._service_role_key = service_role_key
        self._count = 0
        self._cost_usd = Decimal("0")
        self._cost_opg = Decimal("0")
        self._client: AsyncClient | None = None
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False

    @property
    def _enabled(self) -> bool:
        return bool(self._supabase_url and self._service_role_key)

    def start(self) -> None:
        if not self._enabled:
            logger.info("BitQuant OHTTP usage reporting is disabled")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.flush()

    def add(
        self,
        *,
        cost_usd: Decimal | str | float | None,
        cost_opg: Decimal | str | float | None = None,
    ) -> None:
        if not self._enabled:
            return
        self._count += 1
        self._cost_usd += _decimal_or_zero(cost_usd)
        # The normal SDK/LangChain response exposes token usage, but not the
        # x402 settlement cost in OPG, so BitQuant records OPG as zero
        self._cost_opg += _decimal_or_zero(cost_opg)

    async def _run(self) -> None:
        while self._running:
            await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
            await self.flush()

    async def _get_client(self) -> AsyncClient:
        if self._client is None:
            self._client = await acreate_client(
                supabase_url=self._supabase_url,
                supabase_key=self._service_role_key,
                options=AsyncClientOptions(
                    auto_refresh_token=False,
                    persist_session=False,
                ),
            )
        return self._client

    async def flush(self) -> None:
        if not self._enabled or self._count == 0:
            return

        count, cost_usd, cost_opg = self._count, self._cost_usd, self._cost_opg
        self._count = 0
        self._cost_usd = Decimal("0")
        self._cost_opg = Decimal("0")

        try:
            supabase = await self._get_client()
            await supabase.rpc(
                "record_ohttp_usage",
                {
                    "p_request_count": count,
                    "p_cost_usd": str(cost_usd),
                    "p_cost_opg": str(cost_opg),
                },
            ).execute()
        except asyncio.CancelledError:
            self._count += count
            self._cost_usd += cost_usd
            self._cost_opg += cost_opg
            raise
        except Exception:
            logger.exception("Failed to flush BitQuant usage to ohttp_usage_daily")
            self._count += count
            self._cost_usd += cost_usd
            self._cost_opg += cost_opg
