import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
print(enc.max_token_value)
enc = tiktoken.encoding_for_model("gpt-4")
print(enc.max_token_value)
enc = tiktoken.encoding_for_model("gpt-2")
print(enc.max_token_value)