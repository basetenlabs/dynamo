import asyncio

reponses = [
    {"id":"1","choices":[{"index":0,"delta":{"content":"The","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"},
    {"id":"1","choices":[{"index":0,"delta":{"content":" capital","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"},
    {"id":"1","choices":[{"index":0,"delta":{"content":" of","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"},
    {"id":"1","choices":[{"index":0,"delta":{"content":" France","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"},
    {"id":"1","choices":[{"index":0,"delta":{"content":" is","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"},
    {"id":"1","choices":[{"index":0,"delta":{"content":" Paris","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"},
    {"id":"1","choices":[{"index":0,"delta":{"content":".","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"},
] * 10 + [
    {"id":"1","choices":[{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":"stop"}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
]

async def generate(request, context, *args): # generate(request, name: str, is_stopped: callable[bool])
    print("context", context)
    print("id", context.id())
    for response in reponses:
        if context.is_stopped():
            print("request got canceled, canceling child context")
            context.child_stop_generating()
            print("child context canceled")
            break
        yield response
        await asyncio.sleep(0.1)
