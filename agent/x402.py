import httpx
import typing
import logging

from x402.clients.base import x402Client
from x402.types import x402PaymentRequiredResponse, PaymentRequirements


class X402Auth(httpx.Auth):
    """Auth class for handling x402 payment requirements."""

    def __init__(
        self,
        account: typing.Any,
        max_value: typing.Optional[int] = None,
        payment_requirements_selector: typing.Optional[
            typing.Callable[
                [
                    list[PaymentRequirements],
                    typing.Optional[str],
                    typing.Optional[str],
                    typing.Optional[int],
                ],
                PaymentRequirements,
            ]
        ] = None,
    ):
        self.x402_client = x402Client(
            account,
            max_value=max_value,
            payment_requirements_selector=payment_requirements_selector,  # type: ignore
        )

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> typing.AsyncGenerator[httpx.Request, httpx.Response]:
        response = yield request

        if response.status_code == 402:
            try:
                await response.aread()
                data = response.json()

                payment_response = x402PaymentRequiredResponse(**data)

                selected_requirements = self.x402_client.select_payment_requirements(
                    payment_response.accepts
                )

                payment_header = self.x402_client.create_payment_header(
                    selected_requirements, payment_response.x402_version
                )

                request.headers["X-Payment"] = payment_header
                request.headers["Access-Control-Expose-Headers"] = "X-Payment-Response"
                yield request

            except Exception as e:
                logging.error(f"X402Auth: Error handling payment: {e}")
                return
