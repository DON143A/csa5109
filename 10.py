import hashlib

message = "Hello, world!"
result = hashlib.sha1(message.encode())

print("Original Message:", message)
print("SHA-1 Hash:", result.hexdigest())
