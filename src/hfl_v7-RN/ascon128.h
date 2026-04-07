// =====================================================================
// ascon128.h — ASCON-128 Lightweight Authenticated Encryption
// =====================================================================
// NIST LWC Winner 2023 - Optimizado para ESP32-S3
// Spec: https://ascon.iaik.tugraz.at/
// 
// API:
//   ascon128_encrypt(plaintext, pt_len, key, nonce, ciphertext, tag)
//   ascon128_decrypt(ciphertext, ct_len, key, nonce, tag, plaintext) -> bool
// =====================================================================

#ifndef ASCON128_H
#define ASCON128_H

#include <stdint.h>
#include <string.h>

#define ASCON_RATE 8
#define ASCON_KEY_SIZE 16
#define ASCON_NONCE_SIZE 16
#define ASCON_TAG_SIZE 16

// Permutación de estado ASCON
static inline uint64_t ascon_rotr(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

static void ascon_permutation(uint64_t s[5], int rounds) {
    for (int i = 12 - rounds; i < 12; i++) {
        // Addition of constants
        s[2] ^= ((0xfULL - i) << 4) | i;
        
        // Substitution layer
        s[0] ^= s[4];    s[4] ^= s[3];    s[2] ^= s[1];
        uint64_t t[5];
        t[0] = s[0] ^ (~s[1] & s[2]);
        t[1] = s[1] ^ (~s[2] & s[3]);
        t[2] = s[2] ^ (~s[3] & s[4]);
        t[3] = s[3] ^ (~s[4] & s[0]);
        t[4] = s[4] ^ (~s[0] & s[1]);
        t[1] ^= t[0];    t[0] ^= t[4];    t[3] ^= t[2];    t[2] = ~t[2];
        
        // Linear diffusion layer
        s[0] = t[0] ^ ascon_rotr(t[0], 19) ^ ascon_rotr(t[0], 28);
        s[1] = t[1] ^ ascon_rotr(t[1], 61) ^ ascon_rotr(t[1], 39);
        s[2] = t[2] ^ ascon_rotr(t[2],  1) ^ ascon_rotr(t[2],  6);
        s[3] = t[3] ^ ascon_rotr(t[3], 10) ^ ascon_rotr(t[3], 17);
        s[4] = t[4] ^ ascon_rotr(t[4],  7) ^ ascon_rotr(t[4], 41);
    }
}

// Inicialización ASCON-128
static void ascon_init(uint64_t s[5], const uint8_t key[16], const uint8_t nonce[16]) {
    s[0] = 0x80400c0600000000ULL; // IV para ASCON-128
    s[1] = ((uint64_t)key[0] << 56) | ((uint64_t)key[1] << 48) | ((uint64_t)key[2] << 40) | ((uint64_t)key[3] << 32) |
           ((uint64_t)key[4] << 24) | ((uint64_t)key[5] << 16) | ((uint64_t)key[6] << 8)  | ((uint64_t)key[7]);
    s[2] = ((uint64_t)key[8] << 56) | ((uint64_t)key[9] << 48) | ((uint64_t)key[10] << 40) | ((uint64_t)key[11] << 32) |
           ((uint64_t)key[12] << 24) | ((uint64_t)key[13] << 16) | ((uint64_t)key[14] << 8) | ((uint64_t)key[15]);
    s[3] = ((uint64_t)nonce[0] << 56) | ((uint64_t)nonce[1] << 48) | ((uint64_t)nonce[2] << 40) | ((uint64_t)nonce[3] << 32) |
           ((uint64_t)nonce[4] << 24) | ((uint64_t)nonce[5] << 16) | ((uint64_t)nonce[6] << 8)  | ((uint64_t)nonce[7]);
    s[4] = ((uint64_t)nonce[8] << 56) | ((uint64_t)nonce[9] << 48) | ((uint64_t)nonce[10] << 40) | ((uint64_t)nonce[11] << 32) |
           ((uint64_t)nonce[12] << 24) | ((uint64_t)nonce[13] << 16) | ((uint64_t)nonce[14] << 8) | ((uint64_t)nonce[15]);
    
    ascon_permutation(s, 12);
    
    s[3] ^= ((uint64_t)key[0] << 56) | ((uint64_t)key[1] << 48) | ((uint64_t)key[2] << 40) | ((uint64_t)key[3] << 32) |
            ((uint64_t)key[4] << 24) | ((uint64_t)key[5] << 16) | ((uint64_t)key[6] << 8)  | ((uint64_t)key[7]);
    s[4] ^= ((uint64_t)key[8] << 56) | ((uint64_t)key[9] << 48) | ((uint64_t)key[10] << 40) | ((uint64_t)key[11] << 32) |
            ((uint64_t)key[12] << 24) | ((uint64_t)key[13] << 16) | ((uint64_t)key[14] << 8) | ((uint64_t)key[15]);
}

// Cifrado ASCON-128
static void ascon128_encrypt(const uint8_t* plaintext, size_t pt_len, 
                            const uint8_t key[16], const uint8_t nonce[16],
                            uint8_t* ciphertext, uint8_t tag[16]) {
    uint64_t s[5];
    ascon_init(s, key, nonce);
    
    // Associated data = vacío (solo cifrado autenticado, sin AAD)
    s[4] ^= 1ULL; // Domain separation
    
    // Procesamiento de plaintext
    size_t i = 0;
    while (i + ASCON_RATE <= pt_len) {
        uint64_t block = ((uint64_t)plaintext[i+0] << 56) | ((uint64_t)plaintext[i+1] << 48) |
                        ((uint64_t)plaintext[i+2] << 40) | ((uint64_t)plaintext[i+3] << 32) |
                        ((uint64_t)plaintext[i+4] << 24) | ((uint64_t)plaintext[i+5] << 16) |
                        ((uint64_t)plaintext[i+6] << 8)  | ((uint64_t)plaintext[i+7]);
        s[0] ^= block;
        
        for(int j=0; j<8; j++) ciphertext[i+j] = (s[0] >> (56 - j*8)) & 0xff;
        ascon_permutation(s, 6);
        i += ASCON_RATE;
    }
    
    // Bloque final (padding)
    if (i < pt_len) {
        uint64_t block = 0;
        for(size_t j=0; j<(pt_len-i); j++) block |= ((uint64_t)plaintext[i+j]) << (56 - j*8);
        block ^= 0x80ULL << (56 - (pt_len-i)*8);
        s[0] ^= block;
        
        for(size_t j=0; j<(pt_len-i); j++) ciphertext[i+j] = (s[0] >> (56 - j*8)) & 0xff;
    } else {
        s[0] ^= 0x8000000000000000ULL;
    }
    
    // Finalización
    s[1] ^= ((uint64_t)key[0] << 56) | ((uint64_t)key[1] << 48) | ((uint64_t)key[2] << 40) | ((uint64_t)key[3] << 32) |
            ((uint64_t)key[4] << 24) | ((uint64_t)key[5] << 16) | ((uint64_t)key[6] << 8)  | ((uint64_t)key[7]);
    s[2] ^= ((uint64_t)key[8] << 56) | ((uint64_t)key[9] << 48) | ((uint64_t)key[10] << 40) | ((uint64_t)key[11] << 32) |
            ((uint64_t)key[12] << 24) | ((uint64_t)key[13] << 16) | ((uint64_t)key[14] << 8) | ((uint64_t)key[15]);
    
    ascon_permutation(s, 12);
    
    s[3] ^= ((uint64_t)key[0] << 56) | ((uint64_t)key[1] << 48) | ((uint64_t)key[2] << 40) | ((uint64_t)key[3] << 32) |
            ((uint64_t)key[4] << 24) | ((uint64_t)key[5] << 16) | ((uint64_t)key[6] << 8)  | ((uint64_t)key[7]);
    s[4] ^= ((uint64_t)key[8] << 56) | ((uint64_t)key[9] << 48) | ((uint64_t)key[10] << 40) | ((uint64_t)key[11] << 32) |
            ((uint64_t)key[12] << 24) | ((uint64_t)key[13] << 16) | ((uint64_t)key[14] << 8) | ((uint64_t)key[15]);
    
    // Tag (128 bits)
    for(int j=0; j<8; j++) tag[j] = (s[3] >> (56 - j*8)) & 0xff;
    for(int j=0; j<8; j++) tag[8+j] = (s[4] >> (56 - j*8)) & 0xff;
}

// Descifrado ASCON-128
static bool ascon128_decrypt(const uint8_t* ciphertext, size_t ct_len,
                             const uint8_t key[16], const uint8_t nonce[16],
                             const uint8_t tag[16], uint8_t* plaintext) {
    uint64_t s[5];
    ascon_init(s, key, nonce);
    
    s[4] ^= 1ULL;
    
    size_t i = 0;
    while (i + ASCON_RATE <= ct_len) {
        uint64_t ct_block = ((uint64_t)ciphertext[i+0] << 56) | ((uint64_t)ciphertext[i+1] << 48) |
                            ((uint64_t)ciphertext[i+2] << 40) | ((uint64_t)ciphertext[i+3] << 32) |
                            ((uint64_t)ciphertext[i+4] << 24) | ((uint64_t)ciphertext[i+5] << 16) |
                            ((uint64_t)ciphertext[i+6] << 8)  | ((uint64_t)ciphertext[i+7]);
        
        uint64_t pt_block = s[0] ^ ct_block;
        for(int j=0; j<8; j++) plaintext[i+j] = (pt_block >> (56 - j*8)) & 0xff;
        s[0] = ct_block;
        ascon_permutation(s, 6);
        i += ASCON_RATE;
    }
    
    if (i < ct_len) {
        size_t remaining = ct_len - i;
        uint64_t pt_pad = 0;
        for(size_t j=0; j<remaining; j++) {
            uint8_t pt_byte = ((s[0] >> (56 - j*8)) & 0xff) ^ ciphertext[i+j];
            plaintext[i+j] = pt_byte;
            pt_pad |= ((uint64_t)pt_byte) << (56 - j*8);
        }
        pt_pad |= 0x80ULL << (56 - remaining*8);
        s[0] ^= pt_pad;
    } else {
        s[0] ^= 0x8000000000000000ULL;
    }
    
    s[1] ^= ((uint64_t)key[0] << 56) | ((uint64_t)key[1] << 48) | ((uint64_t)key[2] << 40) | ((uint64_t)key[3] << 32) |
            ((uint64_t)key[4] << 24) | ((uint64_t)key[5] << 16) | ((uint64_t)key[6] << 8)  | ((uint64_t)key[7]);
    s[2] ^= ((uint64_t)key[8] << 56) | ((uint64_t)key[9] << 48) | ((uint64_t)key[10] << 40) | ((uint64_t)key[11] << 32) |
            ((uint64_t)key[12] << 24) | ((uint64_t)key[13] << 16) | ((uint64_t)key[14] << 8) | ((uint64_t)key[15]);
    
    ascon_permutation(s, 12);
    
    s[3] ^= ((uint64_t)key[0] << 56) | ((uint64_t)key[1] << 48) | ((uint64_t)key[2] << 40) | ((uint64_t)key[3] << 32) |
            ((uint64_t)key[4] << 24) | ((uint64_t)key[5] << 16) | ((uint64_t)key[6] << 8)  | ((uint64_t)key[7]);
    s[4] ^= ((uint64_t)key[8] << 56) | ((uint64_t)key[9] << 48) | ((uint64_t)key[10] << 40) | ((uint64_t)key[11] << 32) |
            ((uint64_t)key[12] << 24) | ((uint64_t)key[13] << 16) | ((uint64_t)key[14] << 8) | ((uint64_t)key[15]);
    
    // Verificar tag
    for(int j=0; j<8; j++) {
        if(tag[j] != ((s[3] >> (56 - j*8)) & 0xff)) return false;
    }
    for(int j=0; j<8; j++) {
        if(tag[8+j] != ((s[4] >> (56 - j*8)) & 0xff)) return false;
    }
    
    return true;
}

// Helper: Generar nonce incrementable (timestamp + counter)
static void ascon_generate_nonce(uint8_t nonce[16], uint32_t timestamp_ms, uint32_t counter) {
    memset(nonce, 0, 16);
    nonce[0] = (timestamp_ms >> 24) & 0xff;
    nonce[1] = (timestamp_ms >> 16) & 0xff;
    nonce[2] = (timestamp_ms >> 8) & 0xff;
    nonce[3] = timestamp_ms & 0xff;
    nonce[12] = (counter >> 24) & 0xff;
    nonce[13] = (counter >> 16) & 0xff;
    nonce[14] = (counter >> 8) & 0xff;
    nonce[15] = counter & 0xff;
}

#endif // ASCON128_H
