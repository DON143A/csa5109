def generate_key_square(key):
    key = key.upper().replace('J', 'I')  
    result = []
    for char in key:
        if char not in result and char.isalpha():
            result.append(char)
    for char in 'ABCDEFGHIKLMNOPQRSTUVWXYZ':
        if char not in result:
            result.append(char)
    return [result[i:i+5] for i in range(0, 25, 5)]

def find_position(matrix, letter):
    for i, row in enumerate(matrix):
        for j, char in enumerate(row):
            if char == letter:
                return i, j

def prepare_text(text):
    text = text.upper().replace('J', 'I')
    prepared = ''
    i = 0
    while i < len(text):
        char1 = text[i]
        char2 = text[i + 1] if i + 1 < len(text) else 'X'
        if char1 == char2:
            prepared += char1 + 'X'
            i += 1
        else:
            prepared += char1 + char2
            i += 2
    if len(prepared) % 2 != 0:
        prepared += 'X'
    return prepared

def encrypt_pair(pair, matrix):
    row1, col1 = find_position(matrix, pair[0])
    row2, col2 = find_position(matrix, pair[1])
    if row1 == row2:
        return matrix[row1][(col1 + 1) % 5] + matrix[row2][(col2 + 1) % 5]
    elif col1 == col2:
        return matrix[(row1 + 1) % 5][col1] + matrix[(row2 + 1) % 5][col2]
    else:
        return matrix[row1][col2] + matrix[row2][col1]

def playfair_encrypt(plaintext, key):
    matrix = generate_key_square(key)
    prepared_text = prepare_text(''.join(filter(str.isalpha, plaintext)))
    encrypted = ''
    for i in range(0, len(prepared_text), 2):
        encrypted += encrypt_pair(prepared_text[i:i+2], matrix)
    return encrypted

# Example usage
key = "MONARCHY"
plaintext = "INSTRUMENTS"
ciphertext = playfair_encrypt(plaintext, key)

print("Key:", key)
print("Plaintext:", plaintext)
print("Ciphertext:", ciphertext)
