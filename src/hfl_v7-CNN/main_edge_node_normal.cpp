// =====================================================================
// main_edge_node_normal.cpp — ESP32-S3 Edge Node (HFL v7-CNN)
// =====================================================================
// Rol:
// 1. Simulación Interna Benigna: Genera ÚNICAMENTE tráfico Normal.
// 2. Extracción de Features (13 variables).
// 3. Inferencia Local (TinyML) — Motor CNN-1D implementado en C++:
//      Conv1D(32, k=3, same) → ReLU
//      Conv1D(16, k=3, same) → ReLU
//      GlobalAveragePooling1D
//      Dense(8)  → ReLU      ← capa federada (W_dense1)
//      Dense(3)  → Softmax   ← capa federada (W_dense_out)
// 4. Envía Features a la Raspberry Pi (fl/features) — ASCON-128.
// 5. Recibe pesos Dense actualizados (fl/global_model) — ASCON-128.
//
// NOTA: Las capas Conv1D son FIJAS (no se federan).
//       Solo W_dense1 y W_dense_out se actualizan vía FL.
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "model_weights.h"    // Generado por train_3class.py (CNN)
#include "ascon128.h"
#include <math.h>
#include <mbedtls/base64.h>

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48
#endif

// ==========================================
// CONFIG WiFi y MQTT GATEWAY
// ==========================================
const char* STA_SSID = "TP-Link_AADB";           // <-- CAMBIAR
const char* STA_PASS = "55707954";                // <-- CAMBIAR
const char* GATEWAY_MQTT_SERVER = "192.168.1.13"; // <-- IP Raspberry Pi 4
const int   GATEWAY_MQTT_PORT   = 1883;

const char* TOPIC_FEATURES     = "fl/features";
const char* TOPIC_GLOBAL_MODEL = "fl/global_model";
const String CLIENT_ID         = "esp32_edge_normal_1";

// ==========================================
// ARQUITECTURA CNN-1D
// Conv1: 32 filtros, kernel=3, padding=same  (13,1) -> (13,32)
// Conv2: 16 filtros, kernel=3, padding=same  (13,32)-> (13,16)
// GAP:   (13,16) -> (16,)
// Dense1: (16,) -> (8,)   <- federada
// DenseO: (8,)  -> (3,)   <- federada
// ==========================================
constexpr size_t CONV1_FILTERS = 32;
constexpr size_t CONV2_FILTERS = 16;  // = GAP_OUT
constexpr size_t DENSE1_UNITS  = 8;
constexpr size_t OUTPUT_UNITS  = 3;
constexpr size_t KERNEL_SIZE   = 3;

const char* CLASS_NAMES_STR[3] = {"normal", "mqtt_bruteforce", "scan_A"};

// ── Pesos Conv (FIJOS, cargados desde model_weights.h)
// W_conv1[k][in_ch][filter], b_conv1[filter]
// W_conv2[k][in_ch][filter], b_conv2[filter]
// (Se usan directamente desde los arrays del header — no se copian)

// ── Pesos Dense (MUTABLES — actualizados por FL)
float Wd1[CONV2_FILTERS][DENSE1_UNITS];  float bd1[DENSE1_UNITS];
float Wdo[DENSE1_UNITS][OUTPUT_UNITS];   float bdo[OUTPUT_UNITS];

// ── Buffers de activaciones intermedias
float conv1_out[FEATURE_COUNT][CONV1_FILTERS]; // (13, 32)
float conv2_out[FEATURE_COUNT][CONV2_FILTERS]; // (13, 16)
float gap_out[CONV2_FILTERS];                  // (16,)
float dense1_out[DENSE1_UNITS];                // (8,)
float softmax_out[OUTPUT_UNITS];               // (3,)

// ==========================================
// TRACKER DE FLUJO
// ==========================================
uint32_t      brokerGlobalPkts  = 0;
uint32_t      brokerGlobalBytes = 0;
float         brokerPshFlags    = 0;
unsigned long brokerLastPktUs   = 0;
float brokerSumIat=0, brokerSumSqIat=0, brokerMinIat=1e9f, brokerMaxIat=0;
float brokerSumPktLen=0, brokerSumSqPktLen=0, brokerMinPktLen=1e9f, brokerMaxPktLen=0;

WiFiClient   wifiClient;
PubSubClient mqttGateway(wifiClient);
int          totalAlertas    = 0;
unsigned long ledOffTime     = 0;
uint32_t     lastSimulationMs = 0;
uint32_t     msg_counter      = 0;

const uint8_t ASCON_KEY[16] = {
  0xA1,0xB2,0xC3,0xD4,0xE5,0xF6,0x07,0x18,
  0x29,0x3A,0x4B,0x5C,0x6D,0x7E,0x8F,0x90
};

// ==========================================
// UTILIDADES
// ==========================================
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
void setLED(uint8_t r, uint8_t g, uint8_t b) { neopixelWrite(RGB_BUILTIN, r, g, b); }

void resetBrokerFlow() {
  brokerGlobalPkts=0; brokerGlobalBytes=0; brokerPshFlags=0;
  brokerLastPktUs=0;
  brokerSumIat=0; brokerSumSqIat=0; brokerMinIat=1e9f; brokerMaxIat=0;
  brokerSumPktLen=0; brokerSumSqPktLen=0; brokerMinPktLen=1e9f; brokerMaxPktLen=0;
}

void brokerTrackEvent(uint16_t pkt_len, bool is_psh) {
  unsigned long now = micros();
  if (brokerGlobalPkts > 0 && brokerLastPktUs > 0) {
    float iat = (now - brokerLastPktUs) / 1e6f;
    brokerSumIat += iat; brokerSumSqIat += iat*iat;
    if (iat < brokerMinIat) brokerMinIat = iat;
    if (iat > brokerMaxIat) brokerMaxIat = iat;
  }
  brokerGlobalPkts++; brokerGlobalBytes += pkt_len;
  if (is_psh) brokerPshFlags += 1.0f;
  brokerSumPktLen += (float)pkt_len; brokerSumSqPktLen += (float)pkt_len*(float)pkt_len;
  if ((float)pkt_len < brokerMinPktLen) brokerMinPktLen = (float)pkt_len;
  if ((float)pkt_len > brokerMaxPktLen) brokerMaxPktLen = (float)pkt_len;
  brokerLastPktUs = now;
}

void brokerExtractFeatures(float out[FEATURE_COUNT]) {
  float n = (float)brokerGlobalPkts;
  for (size_t i=0; i<FEATURE_COUNT; i++) out[i] = 0.0f;
  if (n < 1.0f) return;
  float mean_pkt = brokerSumPktLen / n;
  float mean_iat = (n > 1) ? brokerSumIat / (n-1.0f) : 0.0f;
  float var_pkt  = (n > 1) ? (brokerSumSqPktLen/n) - (mean_pkt*mean_pkt) : 0.0f;
  float var_iat  = (n > 1) ? (brokerSumSqIat/(n-1.0f)) - (mean_iat*mean_iat) : 0.0f;
  if (var_pkt < 0) var_pkt=0; if (var_iat < 0) var_iat=0;
  out[0]=n; out[1]=mean_iat; out[2]=sqrtf(var_iat);
  out[3]=(n>1)?brokerMinIat:0; out[4]=(n>1)?brokerMaxIat:0;
  out[5]=mean_pkt; out[6]=(float)brokerGlobalBytes;
  out[7]=brokerPshFlags; out[8]=0; out[9]=0;
  out[10]=sqrtf(var_pkt); out[11]=(brokerMinPktLen<1e8f)?brokerMinPktLen:0; out[12]=brokerMaxPktLen;
}

// ==========================================
// MOTOR CNN-1D (TinyML en C++)
// ==========================================
// Conv1D con padding="same":
//   output[t][f] = sum_{k=0}^{KERNEL_SIZE-1} input[t + k - KERNEL_SIZE/2][in_ch] * W[k][in_ch][f] + bias[f]
//   Índices fuera de rango → zero padding.
void conv1d_same(const float input[][1],    // (T, in_ch)
                 size_t T, size_t in_ch,
                 const float W[KERNEL_SIZE][1][CONV1_FILTERS],
                 const float bias[CONV1_FILTERS],
                 size_t out_filters,
                 float output[][CONV1_FILTERS]) {  // (T, out_filters)
  int pad = KERNEL_SIZE / 2;
  for (size_t t = 0; t < T; t++) {
    for (size_t f = 0; f < out_filters; f++) {
      float acc = bias[f];
      for (size_t k = 0; k < KERNEL_SIZE; k++) {
        int src = (int)t + (int)k - pad;
        if (src >= 0 && src < (int)T) {
          for (size_t ic = 0; ic < in_ch; ic++) {
            acc += input[src][ic] * W[k][ic][f];
          }
        }
      }
      output[t][f] = relu(acc);
    }
  }
}

void conv1d_same_2(const float input[][CONV1_FILTERS],
                   size_t T,
                   const float W[KERNEL_SIZE][CONV1_FILTERS][CONV2_FILTERS],
                   const float bias[CONV2_FILTERS],
                   float output[][CONV2_FILTERS]) {
  int pad = KERNEL_SIZE / 2;
  for (size_t t = 0; t < T; t++) {
    for (size_t f = 0; f < CONV2_FILTERS; f++) {
      float acc = bias[f];
      for (size_t k = 0; k < KERNEL_SIZE; k++) {
        int src = (int)t + (int)k - pad;
        if (src >= 0 && src < (int)T) {
          for (size_t ic = 0; ic < CONV1_FILTERS; ic++) {
            acc += input[src][ic] * W[k][ic][f];
          }
        }
      }
      output[t][f] = relu(acc);
    }
  }
}

void initModel() {
  // Copiar capas Dense mutables desde el header (Conv se usa directo)
  for (size_t i=0; i<CONV2_FILTERS; i++)
    for (size_t j=0; j<DENSE1_UNITS; j++) Wd1[i][j] = W_dense1[i][j];
  for (size_t j=0; j<DENSE1_UNITS; j++)  bd1[j] = b_dense1[j];
  for (size_t i=0; i<DENSE1_UNITS; i++)
    for (size_t j=0; j<OUTPUT_UNITS; j++) Wdo[i][j] = W_dense_out[i][j];
  for (size_t j=0; j<OUTPUT_UNITS; j++)  bdo[j] = b_dense_out[j];
  Serial.println("[CNN] Pesos cargados: Conv1(fijo)+Conv2(fijo)+Dense1(FL)+DenseOut(FL)");
}

int predictLocal(const float raw[FEATURE_COUNT], float* confidence) {
  // 1. Normalizar (StandardScaler)
  float x_norm[FEATURE_COUNT][1];
  for (size_t i=0; i<FEATURE_COUNT; i++)
    x_norm[i][0] = (raw[i] - scaler_mean[i]) / scaler_std[i];

  // 2. Conv1D-1 (13,1) -> (13,32)  [pesos fijos del header]
  conv1d_same(x_norm, FEATURE_COUNT, 1, W_conv1, b_conv1, CONV1_FILTERS, conv1_out);

  // 3. Conv1D-2 (13,32) -> (13,16) [pesos fijos del header]
  conv1d_same_2(conv1_out, FEATURE_COUNT, W_conv2, b_conv2, conv2_out);

  // 4. GlobalAveragePooling1D: promedio temporal -> (16,)
  for (size_t f=0; f<CONV2_FILTERS; f++) {
    float s=0;
    for (size_t t=0; t<FEATURE_COUNT; t++) s += conv2_out[t][f];
    gap_out[f] = s / (float)FEATURE_COUNT;
  }

  // 5. Dense1 (16->8) + ReLU  [FEDERADA]
  for (size_t j=0; j<DENSE1_UNITS; j++) {
    float acc = bd1[j];
    for (size_t i=0; i<CONV2_FILTERS; i++) acc += gap_out[i]*Wd1[i][j];
    dense1_out[j] = relu(acc);
  }

  // 6. Dense_out (8->3) + Softmax  [FEDERADA]
  float z[OUTPUT_UNITS];
  for (size_t j=0; j<OUTPUT_UNITS; j++) {
    z[j] = bdo[j];
    for (size_t i=0; i<DENSE1_UNITS; i++) z[j] += dense1_out[i]*Wdo[i][j];
  }
  float maxZ = z[0];
  for (size_t j=1; j<OUTPUT_UNITS; j++) if (z[j]>maxZ) maxZ=z[j];
  float sumE=0;
  for (size_t j=0; j<OUTPUT_UNITS; j++) { softmax_out[j]=expf(z[j]-maxZ); sumE+=softmax_out[j]; }
  int pred=0; float maxP=0;
  for (size_t j=0; j<OUTPUT_UNITS; j++) {
    softmax_out[j]/=sumE;
    if (softmax_out[j]>maxP) { maxP=softmax_out[j]; pred=(int)j; }
  }
  *confidence = maxP;
  return pred;
}

// ==========================================
// SIMULACIÓN TRÁFICO NORMAL
// ==========================================
void simulateSelfTraffic() {
  int pkts = random(4, 9);
  Serial.print("\n--- [SIM] Trafico MQTT Normal ("); Serial.print(pkts); Serial.println(" pkts) ---");
  for (int i=0; i<pkts; i++) {
    uint16_t pkt_len;
    bool psh;
    if (i==0 || (random(100)<40)) { pkt_len=52; psh=false; }
    else                           { pkt_len=random(58,112); psh=(random(100)<60); }
    brokerTrackEvent(pkt_len, psh);
    delayMicroseconds(random(35, 680));
  }
  float features[FEATURE_COUNT];
  brokerExtractFeatures(features);
  resetBrokerFlow();

  // Inferencia
  float conf;
  unsigned long t0 = micros();
  int pred = predictLocal(features, &conf);
  unsigned long inf_us = micros() - t0;

  Serial.print("  [CNN] Prediccion ("); Serial.print(inf_us); Serial.print("us): ");
  for (size_t j=0; j<OUTPUT_UNITS; j++) {
    Serial.print(CLASS_NAMES_STR[j]); Serial.print("="); Serial.print(softmax_out[j]*100,1); Serial.print("% | ");
  }
  Serial.print(" => "); Serial.println(CLASS_NAMES_STR[pred]);

  if (pred != 0) {
    totalAlertas++;
    setLED(pred==1?255:0, 0, pred==2?255:0);
    ledOffTime = millis() + 3000;
  } else {
    setLED(0, 10, 0);
  }
  sendFeaturesToGateway(features);
}

// ==========================================
// ENVÍO DE FEATURES AL GATEWAY (ASCON-128)
// ==========================================
void sendFeaturesToGateway(float features[FEATURE_COUNT]) {
  if (!mqttGateway.connected()) return;

  StaticJsonDocument<512> doc;
  doc["client_id"] = CLIENT_ID;
  JsonArray arr = doc.createNestedArray("features");
  for (size_t i=0; i<FEATURE_COUNT; i++) arr.add(features[i]);

  char json_buf[512];
  size_t json_len = serializeJson(doc, json_buf);

  uint8_t nonce[16];
  ascon_generate_nonce(nonce, millis(), msg_counter++);

  uint8_t ct[512], tag[16];
  unsigned long t0 = micros();
  ascon128_encrypt((uint8_t*)json_buf, json_len, ASCON_KEY, nonce, ct, tag);
  unsigned long enc_us = micros() - t0;

  char ct_b64[1024]; size_t ct_len=0;
  char tag_b64[32];  size_t tag_len=0;
  char nc_b64[32];   size_t nc_len=0;
  mbedtls_base64_encode((unsigned char*)ct_b64, sizeof(ct_b64), &ct_len, ct, json_len);
  mbedtls_base64_encode((unsigned char*)tag_b64,sizeof(tag_b64),&tag_len,tag,16);
  mbedtls_base64_encode((unsigned char*)nc_b64, sizeof(nc_b64), &nc_len, nonce,16);

  StaticJsonDocument<1536> enc_doc;
  enc_doc["ct"]    = String(ct_b64).substring(0,ct_len);
  enc_doc["tag"]   = String(tag_b64).substring(0,tag_len);
  enc_doc["nonce"] = String(nc_b64).substring(0,nc_len);

  char payload[1536];
  size_t plen = serializeJson(enc_doc, payload);
  if (mqttGateway.publish(TOPIC_FEATURES, payload, plen)) {
    Serial.print("  [ASCON ENC] "); Serial.print(enc_us); Serial.print("us | ");
    Serial.print(json_len); Serial.print("B -> "); Serial.print(plen); Serial.println("B");
  }
}

// ==========================================
// RECEPCIÓN MODELO GLOBAL (W_dense1 + W_dense_out)
// ==========================================
void onMqttGatewayCallback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) != TOPIC_GLOBAL_MODEL) return;

  Serial.println("\n[FL] ==============================================");
  Serial.println("[FL] NUEVO MODELO CNN RECIBIDO DEL GATEWAY (RPi)");

  DynamicJsonDocument envelope(12288);
  if (deserializeJson(envelope, payload, length)) { Serial.println("[ERROR] Parse envelope"); return; }

  String ct_b64    = envelope["ct"].as<String>();
  String tag_b64   = envelope["tag"].as<String>();
  String nonce_b64 = envelope["nonce"].as<String>();

  uint8_t ct[8192], tag[16], nonce[16];
  size_t ct_len=0, tag_len2=0, nc_len=0;
  mbedtls_base64_decode(ct,   sizeof(ct),  &ct_len,  (unsigned char*)ct_b64.c_str(),    ct_b64.length());
  mbedtls_base64_decode(tag,  16,          &tag_len2,(unsigned char*)tag_b64.c_str(),   tag_b64.length());
  mbedtls_base64_decode(nonce,16,          &nc_len,  (unsigned char*)nonce_b64.c_str(), nonce_b64.length());

  uint8_t plain[8192];
  unsigned long t0 = micros();
  bool ok = ascon128_decrypt(ct, ct_len, ASCON_KEY, nonce, tag, plain);
  unsigned long dec_us = micros() - t0;

  if (!ok) { Serial.println("[ERROR] ASCON: Tag invalido. Rechazado."); return; }
  Serial.print("[FL] [ASCON DEC] "); Serial.print(dec_us); Serial.print("us | ");
  Serial.print(ct_len); Serial.println("B OK");

  DynamicJsonDocument doc(8192);
  if (deserializeJson(doc, plain, ct_len)) { Serial.println("[ERROR] Parse JSON"); return; }

  // Actualizar W_dense1 (GAP_OUT=16 -> DENSE1=8) — capa federada
  JsonArray wd1arr = doc["W_dense1"].as<JsonArray>();
  JsonArray bd1arr = doc["b_dense1"].as<JsonArray>();
  JsonArray wdoarr = doc["W_dense_out"].as<JsonArray>();
  JsonArray bdoarr = doc["b_dense_out"].as<JsonArray>();

  for (size_t i=0; i<CONV2_FILTERS; i++) {
    JsonArray row = wd1arr[i].as<JsonArray>();
    for (size_t j=0; j<DENSE1_UNITS; j++) Wd1[i][j] = row[j].as<float>();
  }
  for (size_t j=0; j<DENSE1_UNITS; j++)  bd1[j] = bd1arr[j].as<float>();
  for (size_t i=0; i<DENSE1_UNITS; i++) {
    JsonArray row = wdoarr[i].as<JsonArray>();
    for (size_t j=0; j<OUTPUT_UNITS; j++) Wdo[i][j] = row[j].as<float>();
  }
  for (size_t j=0; j<OUTPUT_UNITS; j++)  bdo[j] = bdoarr[j].as<float>();

  Serial.println("[FL] Pesos Dense1+DenseOut CNN actualizados. Inferencia activa.");
  Serial.println("[FL] ==============================================");
  setLED(30,30,0); ledOffTime = millis()+2000;
}

// ==========================================
// SETUP & LOOP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(2000);
  setLED(0,10,0);

  WiFi.mode(WIFI_STA);
  WiFi.begin(STA_SSID, STA_PASS);
  Serial.print("\n[NODE-CNN] Conectando a Wi-Fi...");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\n[NODE-CNN] Conectado.");

  mqttGateway.setServer(GATEWAY_MQTT_SERVER, GATEWAY_MQTT_PORT);
  mqttGateway.setBufferSize(8192);
  mqttGateway.setCallback(onMqttGatewayCallback);

  initModel();
  resetBrokerFlow();
  lastSimulationMs = millis();
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttGateway.connected()) {
      if (mqttGateway.connect(CLIENT_ID.c_str())) {
        mqttGateway.subscribe(TOPIC_GLOBAL_MODEL);
        Serial.println("[MQTT] Suscrito al Gateway (RPi).");
      }
    }
    mqttGateway.loop();
  }

  if (millis() - lastSimulationMs >= 5000) {
    lastSimulationMs = millis();
    simulateSelfTraffic();
  }

  if (ledOffTime>0 && millis()>ledOffTime) { setLED(0,10,0); ledOffTime=0; }
  delay(1);
}
