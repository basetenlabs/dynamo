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

class CancellationWatcher:
    """watches and propagates cancellation to the engine."""
    def __init__(self, py_context):
        self.py_context = py_context
        self.monitoring_task = None
        self._is_stopped = py_context.is_stopped()
        self._registered_callbacks = []
        self._callback_lock = asyncio.Lock()

    async def _monitor_loop(self):
        # This method replicates the behavior of the original is_canceled function
        try:
            # Polls every 15 seconds until the request is stopped or the task is cancelled
            while True:
                self._is_stopped = await self.py_context.stopped(15)    
                if self._is_stopped:
                    async with self._callback_lock:
                        for callback in self._registered_callbacks:
                            try:
                                callback()
                            except Exception as e:
                                # Log the error from the callback
                                print(f"Error in callback for {self.id()}: {e}")
                    break
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Log other unexpected errors in the monitoring loop
            print(f"Error in cancellation _monitor_loop for {self.id()}: {e}")
            # Depending on policy, you might want to re-raise e or handle it

    async def __aenter__(self):
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        return self # Allows using 'as watcher_instance' if needed, though not used here
    
    def is_stopped(self):
        return self._is_stopped
    
    async def register_callback(self, callback):
        """Register a callback to be called when the request is stopped."""
        self._registered_callbacks.append(callback)
        async with self._callback_lock:
            if self._is_stopped:
                # If already stopped, call the callback immediately
                callback()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.monitoring_task:
            if not self.monitoring_task.done():
                self.monitoring_task.cancel()


class MockEngine:
    def __init__(self, model_name):
        self.model_name = model_name
        self.counter = 0

    def generate(self, request, py_context):                   
        self.counter += 1
        id = f"chat-{uuid.uuid4()}"
        created = int(time.time())
        model = self.model_name
        print(f"{created} | Received request: {request}")

        async def generator():
            
            async with CancellationWatcher(py_context) as cancel_task:
                if self.counter % 3 == 0:
                    raise HttpError(415 + self.counter, 'bad luck, your schema got rejected during streaming\n')
                def callback():
                    print(f"detected cancellation in callback for {id}")
                
                await cancel_task.register_callback(callback)
                await asyncio.sleep(5)
                num_chunks = 10
                
                for i in range(num_chunks):
                    if cancel_task.is_stopped():
                        print(f"detected cancellation in generator for {id}")
                        break
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
