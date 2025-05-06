def caesar_cipher(text, shift, mode='encrypt'):
    result = ''
    
    if mode == 'decrypt':
        shift = -shift

    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + shift) % 26 + base
            result += chr(shifted)
        else:
            result += char
    
    return result

plain_text = "Hello, World!"
shift_value = 3

encrypted_text = caesar_cipher(plain_text, shift_value, mode='encrypt')
decrypted_text = caesar_cipher(encrypted_text, shift_value, mode='decrypt')

print("Original:", plain_text)
print("Encrypted:", encrypted_text)
print("Decrypted:", decrypted_text)
