# The MIT License (MIT)
# Copyright © 2025 Quant by OpenGradient

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from pydantic import BaseModel


class QuantQuery(BaseModel):
    """
    A query sent from the validator to the miner.
    """

    query: str
    userID: str
    metadata: dict

    model_config = {"arbitrary_types_allowed": True}


class QuantResponse(BaseModel):
    """
    A response sent from the miner to the validator.
    """

    response: str
    signature: bytes
    proofs: list[bytes]
    metadata: dict

    model_config = {"arbitrary_types_allowed": True}

    def validate(self) -> bool:
        """
        Validates the response proofs as well as the signatures.

        Returns:
            bool: True if the proofs are valid, False otherwise.
        """
        # TODO(developer): Implement validation logic for the proofs
        return True
