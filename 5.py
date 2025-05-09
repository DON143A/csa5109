import math

def encrypt_rail_fence(text, key):
    text = text.replace(" ", "")
    rows = key
    cols = math.ceil(len(text) / rows)
    matrix = [['' for _ in range(cols)] for _ in range(rows)]
    index = 0
    for r in range(rows):
        for c in range(cols):
            if index < len(text):
                matrix[r][c] = text[index]
                index += 1
    cipher = ''
    for c in range(cols):
        for r in range(rows):
            if matrix[r][c] != '':
                cipher += matrix[r][c]
    return cipher

def decrypt_rail_fence(cipher, key):
    rows = key
    cols = math.ceil(len(cipher) / rows)
    matrix = [['' for _ in range(cols)] for _ in range(rows)]
    index = 0
    for c in range(cols):
        for r in range(rows):
            if index < len(cipher):
                matrix[r][c] = cipher[index]
                index += 1
    plain = ''
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] != '':
                plain += matrix[r][c]
    return plain

plain_text = "HELLO WORLD"
key = 3

cipher_text = encrypt_rail_fence(plain_text, key)
decrypted_text = decrypt_rail_fence(cipher_text, key)

print("Plain Text   :", plain_text)
print("Key (Rows)   :", key)
print("Encrypted    :", cipher_text)
print("Decrypted    :", decrypted_text)
