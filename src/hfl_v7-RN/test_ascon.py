#!/usr/bin/env python3
"""
Test de ASCON-128 para verificar cifrado/descifrado
"""
import sys
sys.path.insert(0, '/Users/nicasas/Documents/Tesis/Tesis-Sistemas-HFD-2/src/hfl_v7')

from ascon128 import encrypt, decrypt, generate_nonce

# Clave compartida (misma que en el sistema)
ASCON_KEY = bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
                   0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90])

# Test 1: Mensaje corto
print("Test 1: Mensaje corto")
msg1 = b"Hello ASCON-128"
nonce1 = generate_nonce(1000, 1)
ct1, tag1 = encrypt(msg1, ASCON_KEY, nonce1)
pt1 = decrypt(ct1, ASCON_KEY, nonce1, tag1)
assert pt1 == msg1, "Error en descifrado"
print(f"  ✓ OK: {msg1} -> {len(ct1)} bytes cifrados -> descifrado OK")

# Test 2: JSON típico de features
print("\nTest 2: JSON de features (típico ESP32->RPi)")
import json
features_msg = json.dumps({
    "client_id": "esp32_edge_normal_1",
    "features": [14.0, 0.15, 0.02, 0.12, 0.18, 90.0, 1260.0, 14.0, 0.0, 0.0, 5.0, 85.0, 95.0]
}).encode('utf-8')

nonce2 = generate_nonce(2000, 2)
ct2, tag2 = encrypt(features_msg, ASCON_KEY, nonce2)
pt2 = decrypt(ct2, ASCON_KEY, nonce2, tag2)
assert pt2 == features_msg
recovered = json.loads(pt2.decode('utf-8'))
print(f"  ✓ OK: JSON {len(features_msg)} bytes -> cifrado -> descifrado")
print(f"  client_id recuperado: {recovered['client_id']}")

# Test 3: Tag inválido (debe fallar)
print("\nTest 3: Detección de tag corrupto")
bad_tag = bytearray(tag2)
bad_tag[0] ^= 0xFF
pt3 = decrypt(ct2, ASCON_KEY, nonce2, bytes(bad_tag))
assert pt3 is None, "Debería rechazar tag inválido"
print("  ✓ OK: Tag corrupto detectado y rechazado")

# Test 4: Payload de pesos (típico RPi->PC)
print("\nTest 4: JSON de pesos (típico RPi->PC)")
weights_msg = json.dumps({
    "gateway_id": "gateway_A",
    "num_samples": 25,
    "W3": [[0.1, 0.2], [0.3, 0.4]],
    "b3": [0.5, 0.6],
    "W4": [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
    "b4": [1.3, 1.4, 1.5]
}).encode('utf-8')

nonce4 = generate_nonce(3000, 3)
ct4, tag4 = encrypt(weights_msg, ASCON_KEY, nonce4)
pt4 = decrypt(ct4, ASCON_KEY, nonce4, tag4)
assert pt4 == weights_msg
print(f"  ✓ OK: Payload de pesos {len(weights_msg)} bytes -> cifrado -> descifrado")

print("\n" + "="*60)
print("TODOS LOS TESTS PASARON ✓")
print("ASCON-128 está funcionando correctamente")
print("="*60)
