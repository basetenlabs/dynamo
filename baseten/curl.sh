query_llama_once() {
    local input="$1"
    random_id=$(cat /proc/sys/kernel/random/uuid)
    curl -s -o /dev/null -w "%{http_code}\n" -X POST http://0.0.0.0:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "X-Baseten-Billing-Org-Id: org-123" \
        -H "X-Baseten-Request-Id: $random_id" \
        -H "X-Baseten-Model-Version-ID: model-123" \
        -H "X-Baseten-Priority: 140" \
        -d "{\"model\": \"Llama-3.2-1B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"$input\"}], \"stream\": true}"
}

query_100x() {
    local input="$1"
    for i in {1..100}; do
        query_llama_once "$input" &
    done
    wait
}

query_100x "What is the capital of France?"