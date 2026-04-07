// =====================================================================
// main_edge_node_normal_decision_tree.cpp — ESP32-S3 Edge Node (Decision Tree)
// =====================================================================
// Simula tráfico normal, detecta localmente con el modelo clásico y
// envía features cifradas con ASCON al Gateway.
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "model_weights.h"
#include "ascon128.h"
#include <mbedtls/base64.h>
#include <math.h>

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48
#endif

constexpr size_t FEATURE_COUNT = NUM_FEATURES;
const char* CLASS_NAMES_STR[NUM_CLASSES] = {"normal", "mqtt_bruteforce", "scan_A"};

const char* STA_SSID = "CAMBIAR_WIFI";
const char* STA_PASS = "CAMBIAR_PASSWORD";
const char* GATEWAY_MQTT_SERVER = "192.168.40.124";
const int GATEWAY_MQTT_PORT = 1883;
const char* TOPIC_FEATURES = "fl/features";
const String CLIENT_ID = "esp32_edge_normal_decision_tree";

const uint8_t ASCON_KEY[16] = {
  0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
  0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90
};

WiFiClient wifiClient;
PubSubClient mqttGateway(wifiClient);
uint32_t msg_counter = 0;

void setLED(uint8_t r, uint8_t g, uint8_t b) {
  neopixelWrite(RGB_BUILTIN, r, g, b);
}

void publishEncryptedFeatures(const float features[FEATURE_COUNT]) {
  if (!mqttGateway.connected()) return;

  StaticJsonDocument<512> doc;
  doc["client_id"] = CLIENT_ID;
  JsonArray array = doc.createNestedArray("features");
  for (size_t i = 0; i < FEATURE_COUNT; i++) array.add(features[i]);

  char plain[512];
  size_t plain_len = serializeJson(doc, plain);

  uint8_t nonce[16];
  ascon_generate_nonce(nonce, millis(), msg_counter++);

  uint8_t ciphertext[512];
  uint8_t tag[16];
  ascon128_encrypt((uint8_t*)plain, plain_len, ASCON_KEY, nonce, ciphertext, tag);

  char ct_b64[1024];
  char tag_b64[64];
  char nonce_b64[64];
  size_t ct_len = 0, tag_len = 0, nonce_len = 0;
  mbedtls_base64_encode((unsigned char*)ct_b64, sizeof(ct_b64), &ct_len, ciphertext, plain_len);
  mbedtls_base64_encode((unsigned char*)tag_b64, sizeof(tag_b64), &tag_len, tag, 16);
  mbedtls_base64_encode((unsigned char*)nonce_b64, sizeof(nonce_b64), &nonce_len, nonce, 16);

  StaticJsonDocument<1536> envelope;
  envelope["ct"] = String(ct_b64).substring(0, ct_len);
  envelope["tag"] = String(tag_b64).substring(0, tag_len);
  envelope["nonce"] = String(nonce_b64).substring(0, nonce_len);

  char output[1536];
  size_t out_len = serializeJson(envelope, output);
  mqttGateway.publish(TOPIC_FEATURES, output, out_len);
}

void generateNormalFeatures(float out[FEATURE_COUNT]) {
  float pkts = (float)random(4, 9);
  float meanPkt = (float)random(58, 90);
  out[0] = pkts;
  out[1] = random(30, 680) / 1000000.0f;
  out[2] = random(1, 200) / 1000000.0f;
  out[3] = random(1, 30) / 1000000.0f;
  out[4] = random(50, 900) / 1000000.0f;
  out[5] = meanPkt;
  out[6] = pkts * meanPkt;
  out[7] = (float)random(0, 3);
  out[8] = 0.0f;
  out[9] = 0.0f;
  out[10] = random(0, 60) / 10.0f;
  out[11] = 52.0f;
  out[12] = (float)random(58, 112);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  setLED(0, 10, 0);
  WiFi.begin(STA_SSID, STA_PASS);
  mqttGateway.setServer(GATEWAY_MQTT_SERVER, GATEWAY_MQTT_PORT);
}

void loop() {
  if (WiFi.status() == WL_CONNECTED && !mqttGateway.connected()) {
    mqttGateway.connect(CLIENT_ID.c_str());
  }
  mqttGateway.loop();

  float features[FEATURE_COUNT];
  generateNormalFeatures(features);
  float confidence = 0.0f;
  int pred = classify(features, &confidence);

  Serial.print("[NORMAL][Decision Tree] pred=");
  Serial.print(CLASS_NAMES_STR[pred]);
  Serial.print(" conf=");
  Serial.println(confidence, 4);

  publishEncryptedFeatures(features);
  delay(5000);
}
