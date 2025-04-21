curl -X POST http://0.0.0.0:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Baseten-Billing-Org-Id: org-123" \
  -H "X-Baseten-Request-Id: unique-uuid-v4" \
  -d '{
    "model": "Llama-3.2-1B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ],
    "stream": true
  }' # 