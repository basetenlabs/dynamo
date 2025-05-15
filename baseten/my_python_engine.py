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

class CustomPythonException(Exception):
    """Custom exception for Python engine errors."""
    def __init__(self, http_code, user_facing, user_msg, msg):
        self.http_code = int(http_code)
        self.user_facing = bool(user_facing)
        self.user_msg = str(user_msg)
        self.msg = str(msg)
    
COUNTER = 0    

async def generate(request, context, *args): # generate(request, name: str, is_stopped: callable[bool])
    print("processing id", context.id())
    global COUNTER
    COUNTER += 1
    if COUNTER % 2 == 0:
        raise CustomPythonException(422, True, "This is a custom error message", "This is a custom error message")
    
    for response in reponses:
        if context.is_stopped():
            print("request got canceled, canceling child context")
            context.child_stop_generating()
            print("child context canceled")
            break
        yield response
        await asyncio.sleep(0.1)
        
    print("finished processing id", context.id())
