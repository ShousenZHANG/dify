"""Send legacy moderation payloads through the existing guarded requestor."""

from pydantic import JsonValue

from core.extension.api_based_extension_requestor import APIBasedExtensionPoint, APIBasedExtensionRequestor
from domain.moderation.api_parameters import ModerationInputParams, ModerationOutputParams


def request_inputs(api_endpoint: str, api_key: str, params: ModerationInputParams) -> JsonValue:
    """Serialize input parameters and return JSON, propagating request failures."""
    return APIBasedExtensionRequestor(api_endpoint, api_key).request(
        APIBasedExtensionPoint.APP_MODERATION_INPUT, params.model_dump()
    )


def request_outputs(api_endpoint: str, api_key: str, params: ModerationOutputParams) -> JsonValue:
    """Serialize output parameters and return JSON, propagating request failures."""
    return APIBasedExtensionRequestor(api_endpoint, api_key).request(
        APIBasedExtensionPoint.APP_MODERATION_OUTPUT, params.model_dump()
    )
