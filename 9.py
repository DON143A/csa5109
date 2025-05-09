import hashlib

message = "Hello, world!"
result = hashlib.md5(message.encode())

print("Original Message:", message)
print("MD5 Hash:", result.hexdigest())
