from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad

key = b'8bytekey'
data = b'HELLO WORLD'

cipher = DES.new(key, DES.MODE_ECB)
encrypted = cipher.encrypt(pad(data, DES.block_size))
print("Encrypted:", encrypted)

cipher2 = DES.new(key, DES.MODE_ECB)
decrypted = unpad(cipher2.decrypt(encrypted), DES.block_size)
print("Decrypted:", decrypted.decode())
