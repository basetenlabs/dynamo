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

import asyncio
import time
import uuid

import uvloop

from dynamo.llm import HttpAsyncEngine, HttpService, HttpError
from dynamo.runtime import DistributedRuntime, dynamo_worker

class MockEngine:
    def __init__(self, model_name):
        self.model_name = model_name
        self.counter = 0

    def generate(self, request, py_context):       
        async def is_canceled(id):
            id = id[-4:]
            print(f"checking for cancellation {id}")
            while not await py_context.stopped(2):
                print(f"request not canceled {id}")
            if py_context.is_stopped():
                print(f"request {id} got canceled! ")
            else:
                print(f"request {id} not canceled, surprise..")
            return 
            
        self.counter += 1
        id = f"chat-{uuid.uuid4()}"
        created = int(time.time())
        model = self.model_name
        print(f"{created} | Received request: {request}")

        async def generator():
            cancel_task = asyncio.create_task(is_canceled(id))
            try:
                if self.counter % 3 == 0:
                    raise HttpError(415 + self.counter, 'bad luck, your schema got rejected during streaming\n')
                
                await asyncio.sleep(5)
                num_chunks = 10
                # if self.counter % 2 == 0:
                #     raise HttpError(415 + self.counter, 'bad luck, your schema got rejected')
                for i in range(num_chunks):
                    mock_content = f"chunk{i}"
                    finish_reason = "stop" if (i == num_chunks - 1) else None
                    chunk = {
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": i,
                                "delta": {"role": "assistant", "content": mock_content},
                                "logprobs": None,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    await asyncio.sleep(1)
                    yield chunk
            finally:
                cancel_task.cancel()
                print(f"request {id} done")

        return generator()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    model: str = "mock_model"
    served_model_name: str = "mock_model"

    loop = asyncio.get_running_loop()
    python_engine = MockEngine(model)
    engine = HttpAsyncEngine(python_engine.generate, loop)

    host: str = "localhost"
    port: int = 8000
    service: HttpService = HttpService(port=port)
    service.add_chat_completions_model(served_model_name, engine)

    print("Starting service...")
    shutdown_signal = service.run(runtime.child_token())

    try:
        print(f"Serving endpoint: {host}:{port}/v1/models")
        print(f"Serving endpoint: {host}:{port}/v1/chat/completions")
        print(f"Serving the following models: {service.list_chat_completions_models()}")
        # Block until shutdown signal received
        await shutdown_signal
    except KeyboardInterrupt:
        # TODO: Handle KeyboardInterrupt gracefully in triton_worker
        # TODO: Caught by DistributedRuntime or HttpService, so it's not caught here
        pass
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    finally:
        print("Shutting down worker...")
        runtime.shutdown()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
