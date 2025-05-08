# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import subprocess
from pathlib import Path

from components.processor import Processor
from components.worker import TensorRTLLMWorker
from fastapi import FastAPI
from pydantic import BaseModel

from dynamo import sdk
from dynamo.sdk import depends, service
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)


def get_http_binary_path():
    sdk_path = Path(sdk.__file__)
    binary_path = sdk_path.parent / "cli/bin/http"
    if not binary_path.exists():
        return "http"
    else:
        return str(binary_path)


class FrontendConfig(BaseModel):
    served_model_name: str
    endpoint_chat: str
    endpoint_completions: str
    port: int = 8080


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
    app=FastAPI(title="TensorRT LLM Example"),
)
# todo this should be called ApiServer
class Frontend:
    worker = depends(TensorRTLLMWorker)
    processor = depends(Processor)

    def __init__(self):
        config = ServiceConfig.get_instance()
        frontend_config = FrontendConfig(**config.get("Frontend", {}))

        # Chat/completions Endpoint
        subprocess.run(
            [
                "llmctl",
                "http",
                "remove",
                "chat-models",
                frontend_config.served_model_name,
            ]
        )
        subprocess.run(
            [
                "llmctl",
                "http",
                "add",
                "chat-models",
                frontend_config.served_model_name,
                frontend_config.endpoint_chat,
            ]
        )

        # Completions Endpoint
        subprocess.run(
            [
                "llmctl",
                "http",
                "remove",
                "completions",
                frontend_config.served_model_name,
            ]
        )
        subprocess.run(
            [
                "llmctl",
                "http",
                "add",
                "completions",
                frontend_config.served_model_name,
                frontend_config.endpoint_completions,
            ]
        )

        logger.info("Starting HTTP server")
        http_binary = get_http_binary_path()
        process = subprocess.Popen(
            [http_binary, "-p", str(frontend_config.port)], stdout=None, stderr=None
        )
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
