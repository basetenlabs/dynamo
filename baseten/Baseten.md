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

# Testing with dynamo run

```bash
cargo build --release --features python
cd baseten
/node-storage/cargo-target/release/dynamo-run out=pystr:my_python_engine.py in=http --model-name Llama-3.2-1B-Instruct
```

# Testing the Python Bindings

Setup
```
uv venv
# python3.12
source .venv/bin/activate
cd lib/bindings/python
```

Dev Loop
```
maturin develop --uv 
python ./examples/openai_service/server.py 
```


