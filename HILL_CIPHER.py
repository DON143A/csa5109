import numpy as np

def mod_inverse(a, m):
    a = a % m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    raise ValueError("Modular inverse does not exist")

def create_key_matrix(key):
    key = key.upper().replace(" ", "")
    if len(key) != 9:
        raise ValueError("Key must be exactly 9 letters for 3x3 Hill Cipher.")
    key_matrix = [ord(char) - ord('A') for char in key]
    return np.array(key_matrix).reshape(3, 3)


def process_text(text):
    text = text.upper().replace(" ", "")
    while len(text) % 3 != 0:
        text += 'X'
    return text


def hill_encrypt(text, key):
    key_matrix = create_key_matrix(key)
    text = process_text(text)
    ciphertext = ''
    for i in range(0, len(text), 3):
        block = [ord(char) - ord('A') for char in text[i:i+3]]
        result = np.dot(key_matrix, block) % 26
        ciphertext += ''.join([chr(num + ord('A')) for num in result])
    return ciphertext

def hill_decrypt(ciphertext, key):
    key_matrix = create_key_matrix(key)
    det = int(round(np.linalg.det(key_matrix))) % 26

    try:
        det_inv = mod_inverse(det, 26)
    except ValueError:
        raise ValueError("Key matrix is not invertible modulo 26.")

    
    cofactors = np.zeros((3, 3))
    for row in range(3):
        for col in range(3):
            minor = np.delete(np.delete(key_matrix, row, axis=0), col, axis=1)
            cofactors[row][col] = ((-1)**(row+col)) * int(round(np.linalg.det(minor)))
    adjugate = np.transpose(cofactors)
    inverse_matrix = (det_inv * adjugate) % 26
    inverse_matrix = inverse_matrix.astype(int)

    plaintext = ''
    for i in range(0, len(ciphertext), 3):
        block = [ord(char) - ord('A') for char in ciphertext[i:i+3]]
        result = np.dot(inverse_matrix, block) % 26
        plaintext += ''.join([chr(num + ord('A')) for num in result])
    return plaintext

key = "GYBNQKURP"   
plaintext = "ACT"

ciphertext = hill_encrypt(plaintext, key)
decrypted = hill_decrypt(ciphertext, key)

print("Key:", key)
print("Plaintext:", plaintext)
print("Encrypted:", ciphertext)
print("Decrypted:", decrypted)
