# Build main project from source

```
# make sure your default is ubuntu24.04 / py312
export PYO3_PYTHON=/usr/bin/python3
cargo build --release
```

# Cargo on Dev-Cluster

```toml  ~/.cargo/config.toml
[net]
retry = 3
git-fetch-with-cli = true

# pin to non nfs storage to avoid cargo race conditions.
[build]
target-dir = "/node-storage/cargo-target"
```

# Testing locally with dynamo run and Sglang
```
/node-storage/cargo-target/release/dynamo-run in=http out=sglang --model-path unsloth/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --model-name unsloth/Llama-3.2-1B-Instruct
```

```
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
    ]
  }'
```

# Testing with dynamo run

```bash
cargo build --release --features python
cd baseten
/node-storage/cargo-target/debug/dynamo-run out=pystr:my_python_engine.py in=http --model-name Llama-3.2-1B-Instruct
```