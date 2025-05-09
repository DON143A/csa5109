

def generate_key(text, key):
    key = list(key)
    if len(text) == len(key):
        return "".join(key)
    else:
        for i in range(len(text) - len(key)):
            key.append(key[i % len(key)])
    return "".join(key)

def encrypt(plain_text, key):
    cipher_text = []
    key = generate_key(plain_text, key)
    for i in range(len(plain_text)):
        if plain_text[i].isalpha():
            shift = (ord(plain_text[i].upper()) + ord(key[i].upper())) % 26
            cipher_char = chr(shift + ord('A'))
            cipher_text.append(cipher_char)
        else:
            cipher_text.append(plain_text[i])
    return "".join(cipher_text)

def decrypt(cipher_text, key):
    orig_text = []
    key = generate_key(cipher_text, key)
    for i in range(len(cipher_text)):
        if cipher_text[i].isalpha():
            shift = (ord(cipher_text[i].upper()) - ord(key[i].upper()) + 26) % 26
            plain_char = chr(shift + ord('A'))
            orig_text.append(plain_char)
        else:
            orig_text.append(cipher_text[i])
    return "".join(orig_text)

plain = "HELLO WORLD"
key = "KEY"
encrypted = encrypt(plain, key)
decrypted = decrypt(encrypted, key)

print("Plain Text   :", plain)
print("Key          :", key)
print("Encrypted    :", encrypted)
print("Decrypted    :", decrypted)
