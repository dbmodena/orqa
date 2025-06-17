import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("ChatGPT is amazing!")
print(tokens)
print("Token count:", len(tokens))
