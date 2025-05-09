from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives import hashes

private_key = dsa.generate_private_key(key_size=1024)
public_key = private_key.public_key()

message = b"Digital Signature Example"
signature = private_key.sign(message, hashes.SHA256())

try:
    public_key.verify(signature, message, hashes.SHA256())
    print("Signature is valid.")
except:
    print("Signature is invalid.")
