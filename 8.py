P = 23
G = 5

a = 6
A = (G ** a) % P

b = 15
B = (G ** b) % P

shared_key_alice = (B ** a) % P
shared_key_bob = (A ** b) % P

print(f"P: {P}")
print(f"G: {G}")
print(f"Alice's Public Key: {A}")
print(f"Bob's Public Key: {B}")
print(f"Alice's Shared Key: {shared_key_alice}")
print(f"Bob's Shared Key: {shared_key_bob}")
print("Key Exchange Successful:", shared_key_alice == shared_key_bob)
