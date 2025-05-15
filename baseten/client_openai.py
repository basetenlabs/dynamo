# query_llama_once() {
#     local input="$1"
#     random_id=$(cat /proc/sys/kernel/random/uuid)
#     curl -s -o /dev/null -w "%{http_code}\n" -X POST http://0.0.0.0:8080/v1/chat/completions \
#         -H "Content-Type: application/json" \
#         -H "X-Baseten-Billing-Org-Id: org-123" \
#         -H "X-Baseten-Request-Id: $random_id" \
#         -H "X-Baseten-Model-Version-ID: model-123" \
#         -H "X-Baseten-Priority: 140" \
#         -d "{\"model\": \"Llama-3.2-1B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"$input\"}], \"stream\": true}"
# }

import requests
import json
import random 
def query_llama_once(input_text):
    url = "http://0.0.0.0:8080/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Baseten-Billing-Org-Id": "org-123",
        "X-Baseten-Request-Id": str(random.randint(1, 1000000)),  # Random ID for each request
        "X-Baseten-Model-Version-ID": "model-123",
        "X-Baseten-Priority": "140"
    }
    data = {
        "model": "Llama-3.2-1B-Instruct",
        "messages": [{"role": "user", "content": input_text}],
        "stream": True
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

if __name__ == "__main__":
    input_text = "What is the capital of France?"
    query_llama_once(input_text)