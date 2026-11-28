from typing import Any, override

from core.moderation.base import (
    ApiModerationRuntime,
    Moderation,
    ModerationAction,
    ModerationInputsResult,
    ModerationOutputsResult,
)
from core.moderation.factory import get_api_moderation
from domain.moderation.api_parameters import ModerationInputParams, ModerationOutputParams


class ApiModeration(Moderation):
    name: str = "api"
    _runtime: ApiModerationRuntime | None = None

    @classmethod
    @override
    def validate_config(cls, tenant_id: str, config: dict[str, Any]):
        """
        Validate the incoming form config data.

        :param tenant_id: the id of workspace
        :param config: the form config data
        :return:
        """
        cls._validate_inputs_and_outputs_config(config, False)

        api_based_extension_id = config.get("api_based_extension_id")
        if not api_based_extension_id:
            raise ValueError("api_based_extension_id is required")

        get_api_moderation().validate_extension(tenant_id, api_based_extension_id)

    @override
    def moderation_for_inputs(self, inputs: dict[str, Any], query: str = "") -> ModerationInputsResult:
        flagged = False
        preset_response = ""
        if self.config is None:
            raise ValueError("The config is not set.")

        if self.config["inputs_config"]["enabled"]:
            params = ModerationInputParams(app_id=self.app_id, inputs=inputs, query=query)

            result = self._get_runtime().request_inputs(
                self.tenant_id,
                self.config.get("api_based_extension_id", ""),
                params,
            )
            return ModerationInputsResult.model_validate(result)

        return ModerationInputsResult(
            flagged=flagged, action=ModerationAction.DIRECT_OUTPUT, preset_response=preset_response
        )

    @override
    def moderation_for_outputs(self, text: str) -> ModerationOutputsResult:
        flagged = False
        preset_response = ""
        if self.config is None:
            raise ValueError("The config is not set.")

        if self.config["outputs_config"]["enabled"]:
            params = ModerationOutputParams(app_id=self.app_id, text=text)

            result = self._get_runtime().request_outputs(
                self.tenant_id,
                self.config.get("api_based_extension_id", ""),
                params,
            )
            return ModerationOutputsResult.model_validate(result)

        return ModerationOutputsResult(
            flagged=flagged, action=ModerationAction.DIRECT_OUTPUT, preset_response=preset_response
        )

    def _get_runtime(self) -> ApiModerationRuntime:
        """Bind on the first enabled request and preserve that runtime across contexts.

        A missing or invalid application binding raises before external work.
        """
        if self._runtime is None:
            self._runtime = get_api_moderation()
        return self._runtime
