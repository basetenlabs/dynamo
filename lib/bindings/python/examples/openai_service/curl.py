# #!/bin/bash
# # SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# # list models
# echo "\n\n### Listing models"
# curl http://localhost:8000/v1/models

# # create completion
# echo "\n\n### Creating completions\n"


# curl -X POST http://localhost:8000/v1/chat/completions \
# -H 'accept: application/json' \
# -H 'Content-Type: application/json' \
# -d '{
#     "model": "mock_model",
#     "messages": [
#       {
#         "role":"user",
#         "content":"Hello! How are you?"
#       }
#     ],
#     "max_tokens": 64,
#     "stream": true,
#     "temperature": 0.7,
#     "top_p": 0.9,
#     "frequency_penalty": 0.1,
#     "presence_penalty": 0.2,
#     "top_k": 5
#   }'

# echo "\n"

# Script as python

import requests
import json
import random

print("### Listing models")
response = requests.get("http://localhost:8000/v1/models")
if response.status_code == 200:
    print("Models:", response.json())
else:
    print("Error:", response.status_code, response.text)

print("\n\n### Creating completions")
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
data = {
    "model": "mock_model",
    "messages": [
        {
            "role": "user",
            "content": "Hello! How are you?"
        }
    ],
    "max_tokens": 64,
    "stream": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.2,
    "top_k": 5
}
response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
if response.status_code == 200:
    response_agg = ""
    for line in response.iter_lines():
        if line and not line.startswith(b"data: [DONE]"):
            try:
                json_line = json.loads(line)
                response_agg += json_line["choices"][0]["delta"]["content"]
            except json.JSONDecodeError:
                print("Error decoding JSON:", line)
    print("Response:", response_agg)
else:
    print("Error:", response.status_code, response.text)