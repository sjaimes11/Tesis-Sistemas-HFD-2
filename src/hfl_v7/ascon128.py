"""
=============================================================================
 ascon128.py — ASCON-128 Lightweight Authenticated Encryption (Python)
=============================================================================
 NIST LWC Winner 2023
 Spec: https://ascon.iaik.tugraz.at/
 
 API:
   encrypt(plaintext: bytes, key: bytes, nonce: bytes) -> (ciphertext, tag)
   decrypt(ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes) -> plaintext or None
=============================================================================
"""

ASCON_RATE = 8
ASCON_KEY_SIZE = 16
ASCON_NONCE_SIZE = 16
ASCON_TAG_SIZE = 16

def rotr(val, r):
    return ((val >> r) | (val << (64 - r))) & 0xFFFFFFFFFFFFFFFF

def ascon_permutation(s, rounds):
    M = 0xFFFFFFFFFFFFFFFF
    for i in range(12 - rounds, 12):
        s[2] ^= ((0xf - i) << 4) | i
        
        s[0] ^= s[4]; s[4] ^= s[3]; s[2] ^= s[1]
        t0 = (s[0] ^ (~s[1] & s[2])) & M
        t1 = (s[1] ^ (~s[2] & s[3])) & M
        t2 = (s[2] ^ (~s[3] & s[4])) & M
        t3 = (s[3] ^ (~s[4] & s[0])) & M
        t4 = (s[4] ^ (~s[0] & s[1])) & M
        t1 ^= t0; t0 ^= t4; t3 ^= t2; t2 = (~t2) & M
        
        s[0] = (t0 ^ rotr(t0, 19) ^ rotr(t0, 28)) & M
        s[1] = (t1 ^ rotr(t1, 61) ^ rotr(t1, 39)) & M
        s[2] = (t2 ^ rotr(t2, 1) ^ rotr(t2, 6)) & M
        s[3] = (t3 ^ rotr(t3, 10) ^ rotr(t3, 17)) & M
        s[4] = (t4 ^ rotr(t4, 7) ^ rotr(t4, 41)) & M

def bytes_to_u64(b):
    return int.from_bytes(b, byteorder='big')

def u64_to_bytes(x):
    return x.to_bytes(8, byteorder='big')

def ascon_init(key, nonce):
    s = [0] * 5
    s[0] = 0x80400c0600000000
    s[1] = bytes_to_u64(key[:8])
    s[2] = bytes_to_u64(key[8:16])
    s[3] = bytes_to_u64(nonce[:8])
    s[4] = bytes_to_u64(nonce[8:16])
    
    ascon_permutation(s, 12)
    
    s[3] ^= bytes_to_u64(key[:8])
    s[4] ^= bytes_to_u64(key[8:16])
    
    return s

def encrypt(plaintext, key, nonce):
    assert len(key) == ASCON_KEY_SIZE
    assert len(nonce) == ASCON_NONCE_SIZE
    
    s = ascon_init(key, nonce)
    s[4] ^= 1
    
    ciphertext = bytearray()
    i = 0
    
    while i + ASCON_RATE <= len(plaintext):
        block = bytes_to_u64(plaintext[i:i+8])
        s[0] ^= block
        ciphertext.extend(u64_to_bytes(s[0]))
        ascon_permutation(s, 6)
        i += ASCON_RATE
    
    if i < len(plaintext):
        remaining = plaintext[i:]
        s[0] ^= bytes_to_u64(remaining + b'\x80' + b'\x00' * (7 - len(remaining)))
        ciphertext.extend(u64_to_bytes(s[0])[:len(remaining)])
    else:
        s[0] ^= 0x8000000000000000
    
    s[1] ^= bytes_to_u64(key[:8])
    s[2] ^= bytes_to_u64(key[8:16])
    ascon_permutation(s, 12)
    s[3] ^= bytes_to_u64(key[:8])
    s[4] ^= bytes_to_u64(key[8:16])
    
    tag = u64_to_bytes(s[3]) + u64_to_bytes(s[4])
    
    return bytes(ciphertext), tag

def decrypt(ciphertext, key, nonce, tag):
    assert len(key) == ASCON_KEY_SIZE
    assert len(nonce) == ASCON_NONCE_SIZE
    assert len(tag) == ASCON_TAG_SIZE
    
    s = ascon_init(key, nonce)
    s[4] ^= 1
    
    plaintext = bytearray()
    i = 0
    
    while i + ASCON_RATE <= len(ciphertext):
        ct_block = bytes_to_u64(ciphertext[i:i+8])
        plaintext.extend(u64_to_bytes(s[0] ^ ct_block))
        s[0] = ct_block
        ascon_permutation(s, 6)
        i += ASCON_RATE
    
    if i < len(ciphertext):
        remaining = ciphertext[i:]
        pt_partial = bytearray()
        for j in range(len(remaining)):
            pt_byte = ((s[0] >> (56 - j*8)) & 0xff) ^ remaining[j]
            pt_partial.append(pt_byte)
        plaintext.extend(pt_partial)
        
        s[0] ^= bytes_to_u64(bytes(pt_partial) + b'\x80' + b'\x00' * (7 - len(remaining)))
    else:
        s[0] ^= 0x8000000000000000
    
    s[1] ^= bytes_to_u64(key[:8])
    s[2] ^= bytes_to_u64(key[8:16])
    ascon_permutation(s, 12)
    s[3] ^= bytes_to_u64(key[:8])
    s[4] ^= bytes_to_u64(key[8:16])
    
    computed_tag = u64_to_bytes(s[3]) + u64_to_bytes(s[4])
    
    if computed_tag != tag:
        return None
    
    return bytes(plaintext)

def generate_nonce(timestamp_ms, counter):
    nonce = bytearray(16)
    nonce[0:4] = (timestamp_ms & 0xFFFFFFFF).to_bytes(4, byteorder='big')
    nonce[12:16] = (counter & 0xFFFFFFFF).to_bytes(4, byteorder='big')
    return bytes(nonce)
