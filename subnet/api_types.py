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

import typing
import bittensor as bt
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


class QuantSynapse(bt.Synapse):
    """
    A simple protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    Attributes:
    - query: A QuantQuery object representing the input request sent by the validator.
    - response: An optional QuantResponse object which, when filled, represents the response from the miner.
    """
    
    # Add model configuration to allow arbitrary types
    model_config = {"arbitrary_types_allowed": True}
    
    # Required request input, filled by sending dendrite caller.
    query: typing.Optional[QuantQuery] = None

    # Optional request output, filled by receiving axon.
    response: typing.Optional[QuantResponse] = None

    def __init__(self, query: typing.Optional[QuantQuery] = None, **kwargs):
        """
        Initializes the QuantSynapse with the given parameters.
        This handles either direct instantiation with query or through unpacking attributes.
        """
        # Call the parent class constructor with all kwargs except query
        new_kwargs = {k: v for k, v in kwargs.items() if k != 'query'}
        super().__init__(**new_kwargs)
        
        # Set the query if provided
        if query is not None:
            self.query = query


    def deserialize(self) -> QuantResponse:
        """
        Deserialize the response. This method retrieves the response from
        the miner in the form of response, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - QuantResponse: The deserialized response, which in this case is the value of response.

        Example:
        Assuming a QuantSynapse instance has a response value filled:
        >>> synapse_instance = QuantSynapse(query=QuantQuery("example", "0x123", {}))
        >>> synapse_instance.response = QuantResponse("response_data", b'signature', [b'proof1'], {})
        >>> synapse_instance.deserialize()
        QuantResponse("response_data", b'signature', [b'proof1'], {})
        """
        return self.response