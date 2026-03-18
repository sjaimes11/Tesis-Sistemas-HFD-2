// =====================================================================
// main_broker_hfl.cpp — ESP32-S3 Broker + IDS 3-Class + HFL v6
// =====================================================================
// Modelo: 13 -> 32 -> 16 -> 8 -> 3 (softmax)
// Clases: normal(0), mqtt_bruteforce(1), scan_A(2)
// FL comparte: W3(16,8) b3(8) W4(8,3) b4(3)
// Bidireccional: recibe modelo global del PC via Raspberry Pi
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <sMQTTBroker.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>
#include "model_weights.h"

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48
#endif

// ==========================================
// CONFIG WiFi
// ==========================================
const char* AP_SSID = "FL_BROKER_NET";
const char* AP_PASS = "federated123";
const char* STA_SSID = "TP-Link_AADB";       // <-- CAMBIAR
const char* STA_PASS = "55707954";            // <-- CAMBIAR
const char* RASP_MQTT_SERVER = "192.168.1.13"; // <-- IP Raspberry
const int RASP_MQTT_PORT = 1883;

// ==========================================
// TOPICS MQTT
// ==========================================
const char* TOPIC_FEATURES     = "fl/features";
const char* TOPIC_ALERTS       = "fl/alerts";
const char* TOPIC_UPDATE       = "fl/updates";
const char* TOPIC_GLOBAL_MODEL = "fl/global_model";
const String CLIENT_ID = "esp32_broker_01";

// ==========================================
// MODELO MLP: 13 -> 32 -> 16 -> 8 -> 3
// ==========================================
constexpr size_t L1_UNITS = 32;
constexpr size_t L2_UNITS = 16;
constexpr size_t L3_UNITS = 8;
constexpr size_t OUTPUT_UNITS = NUM_CLASSES; // 3

const float ATTACK_THRESHOLD = 0.5f; // softmax: max class > 0.5
const float LEARNING_RATE = 0.001f;
constexpr uint32_t MIN_PKTS_FOR_ML = 10;
constexpr uint32_t RULE_PKTS_ALERT = 200;

int currentRound = 0;
int numSamples = 0;
constexpr int SAMPLES_PER_UPDATE = 5;

const char* CLASS_NAMES_STR[3] = {"normal", "mqtt_bruteforce", "scan_A"};

// Pesos mutables
float W1[FEATURE_COUNT][L1_UNITS];
float b1[L1_UNITS];
float W2[L1_UNITS][L2_UNITS];
float b2[L2_UNITS];
float W3[L2_UNITS][L3_UNITS];
float b3[L3_UNITS];
float W4[L3_UNITS][OUTPUT_UNITS];
float b4[OUTPUT_UNITS];

// Activaciones
float a0[FEATURE_COUNT];
float z1[L1_UNITS], a1[L1_UNITS];
float z2[L2_UNITS], a2[L2_UNITS];
float z3[L3_UNITS], a3_val[L3_UNITS];
float z4[OUTPUT_UNITS], a4[OUTPUT_UNITS]; // softmax output

// Ventana de monitoring
constexpr uint32_t BROKER_WINDOW_MS = 5000;
uint32_t brokerGlobalPkts = 0;
uint32_t brokerGlobalBytes = 0;
uint32_t brokerConnections = 0;
unsigned long brokerLastWindowMs = 0;
unsigned long brokerFirstPktUs = 0;
unsigned long brokerLastPktUs = 0;
float brokerSumIat = 0, brokerSumSqIat = 0;
float brokerMinIat = 1e9f, brokerMaxIat = 0;
float brokerSumPktLen = 0, brokerSumSqPktLen = 0;
float brokerMinPktLen = 1e9f, brokerMaxPktLen = 0;

WiFiClient raspClient;
PubSubClient mqttRasp(raspClient);

int totalAlertas = 0;
unsigned long ledOffTime = 0;

// Forward declarations
void processReceivedFeatures(const char* payload, unsigned int length);
void sendDeltasToRaspberry();
void analyzeAndAlert(float features[FEATURE_COUNT], const char* source, bool ruleTriggered);
void onMqttRaspCallback(char* topic, byte* payload, unsigned int length);

// ==========================================
// HEURISTIC LABELING (3 clases)
// ==========================================
int heuristicLabel(float features[FEATURE_COUNT], bool ruleTriggered) {
    uint32_t pkts    = (uint32_t)features[0];
    float    meanIat = features[1];
    float    numPsh  = features[7];

    // Regla: muchos paquetes con IAT muy chico y PSH flags → bruteforce
    if (ruleTriggered || pkts >= 100) {
        if (numPsh > 5.0f) return 1;  // mqtt_bruteforce (PSH flags)
        return 2;  // scan_A (puro flood sin PSH)
    }
    // Tráfico lento y con pocos paquetes → normal
    if (pkts <= 30 && meanIat >= 0.1f) return 0;
    // IAT muy bajo y muchos paquetes → scan
    if (meanIat > 0 && meanIat <= 0.01f && pkts > MIN_PKTS_FOR_ML) return 2;
    return -1; // indeterminado
}

// ==========================================
// BROKER FLOW TRACKING
// ==========================================
void resetBrokerFlow() {
  brokerGlobalPkts = 0; brokerGlobalBytes = 0; brokerConnections = 0;
  brokerFirstPktUs = 0; brokerLastPktUs = 0;
  brokerSumIat = 0; brokerSumSqIat = 0; brokerMinIat = 1e9f; brokerMaxIat = 0;
  brokerSumPktLen = 0; brokerSumSqPktLen = 0; brokerMinPktLen = 1e9f; brokerMaxPktLen = 0;
}

void brokerTrackEvent(uint16_t pkt_len) {
  unsigned long now = micros();
  if (brokerGlobalPkts > 0 && brokerLastPktUs > 0) {
    float iat = (now - brokerLastPktUs) / 1e6f;
    brokerSumIat += iat; brokerSumSqIat += iat * iat;
    if (iat < brokerMinIat) brokerMinIat = iat;
    if (iat > brokerMaxIat) brokerMaxIat = iat;
  } else { brokerFirstPktUs = now; }
  brokerGlobalPkts++; brokerGlobalBytes += pkt_len;
  brokerSumPktLen += (float)pkt_len; brokerSumSqPktLen += (float)pkt_len * pkt_len;
  if ((float)pkt_len < brokerMinPktLen) brokerMinPktLen = (float)pkt_len;
  if ((float)pkt_len > brokerMaxPktLen) brokerMaxPktLen = (float)pkt_len;
  brokerLastPktUs = now;
}

void brokerExtractFeatures(float out[FEATURE_COUNT]) {
  float n = (float)brokerGlobalPkts;
  for (size_t i = 0; i < FEATURE_COUNT; i++) out[i] = 0.0f;
  if (n < 1.0f) return;
  float mean_pkt = brokerSumPktLen / n;
  float mean_iat = (n > 1) ? brokerSumIat / (n - 1.0f) : 0.0f;
  float var_pkt = (n > 1) ? (brokerSumSqPktLen / n) - (mean_pkt * mean_pkt) : 0.0f;
  float var_iat = (n > 1) ? (brokerSumSqIat / (n - 1.0f)) - (mean_iat * mean_iat) : 0.0f;
  if (var_pkt < 0) var_pkt = 0; if (var_iat < 0) var_iat = 0;
  out[0]=n; out[1]=mean_iat; out[2]=sqrtf(var_iat);
  out[3]=(n>1)?brokerMinIat:0; out[4]=(n>1)?brokerMaxIat:0;
  out[5]=mean_pkt; out[6]=(float)brokerGlobalBytes;
  out[7]=0; out[8]=0; out[9]=0;
  out[10]=sqrtf(var_pkt);
  out[11]=(brokerMinPktLen<1e8f)?brokerMinPktLen:0;
  out[12]=brokerMaxPktLen;
}

// ==========================================
// BROKER MQTT EMBEBIDO
// ==========================================
class MyBroker : public sMQTTBroker {
public:
  bool onEvent(sMQTTEvent *event) override {
    switch (event->Type()) {
      case NewClient_sMQTTEventType: {
        brokerConnections++;
        brokerTrackEvent(64);
        if (brokerConnections <= 3 || brokerConnections % 50 == 0) {
          sMQTTNewClientEvent *e = (sMQTTNewClientEvent*)event;
          Serial.print("[BROKER] Cliente #"); Serial.print(brokerConnections);
          Serial.print(": "); Serial.println(e->Client()->getClientId().c_str());
        }
        break;
      }
      case LostConnect_sMQTTEventType: { brokerTrackEvent(32); break; }
      case Subscribe_sMQTTEventType: { brokerTrackEvent(48); break; }
      case Public_sMQTTEventType: {
        sMQTTPublicClientEvent *e = (sMQTTPublicClientEvent*)event;
        String topic = e->Topic().c_str();
        String payload = e->Payload().c_str();
        uint16_t msgSize = (uint16_t)(topic.length() + payload.length() + 8);
        brokerTrackEvent(msgSize);
        if (topic == TOPIC_FEATURES) {
          processReceivedFeatures(payload.c_str(), payload.length());
        }
        break;
      }
      default: break;
    }
    return true;
  }
};

MyBroker myBroker;

// ==========================================
// INIT MODEL
// ==========================================
void initModel() {
  for (size_t i = 0; i < FEATURE_COUNT; i++)
    for (size_t j = 0; j < L1_UNITS; j++) W1[i][j] = W1_base[i][j];
  for (size_t j = 0; j < L1_UNITS; j++) b1[j] = b1_base[j];
  for (size_t i = 0; i < L1_UNITS; i++)
    for (size_t j = 0; j < L2_UNITS; j++) W2[i][j] = W2_base[i][j];
  for (size_t j = 0; j < L2_UNITS; j++) b2[j] = b2_base[j];
  for (size_t i = 0; i < L2_UNITS; i++)
    for (size_t j = 0; j < L3_UNITS; j++) W3[i][j] = W3_base[i][j];
  for (size_t j = 0; j < L3_UNITS; j++) b3[j] = b3_base[j];
  for (size_t i = 0; i < L3_UNITS; i++)
    for (size_t j = 0; j < OUTPUT_UNITS; j++) W4[i][j] = W4_base[i][j];
  for (size_t j = 0; j < OUTPUT_UNITS; j++) b4[j] = b4_base[j];
  Serial.println("[MODEL] 13->32->16->8->3 cargado (softmax).");
}

void setLED(uint8_t r, uint8_t g, uint8_t b) { neopixelWrite(RGB_BUILTIN, r, g, b); }

// ==========================================
// MATEMATICAS
// ==========================================
float relu(float x) { return x > 0 ? x : 0.0f; }
float relu_deriv(float x) { return x > 0 ? 1.0f : 0.0f; }

// ==========================================
// FORWARD PASS (softmax)
// ==========================================
int predictLocal(const float raw_features[FEATURE_COUNT], float* confidence) {
  // Normalizar
  for (size_t i = 0; i < FEATURE_COUNT; i++)
    a0[i] = (raw_features[i] - scaler_mean[i]) / scaler_std[i];

  // Capa 1
  for (size_t j = 0; j < L1_UNITS; j++) {
    z1[j] = b1[j];
    for (size_t i = 0; i < FEATURE_COUNT; i++) z1[j] += a0[i] * W1[i][j];
    a1[j] = relu(z1[j]);
  }
  // Capa 2
  for (size_t j = 0; j < L2_UNITS; j++) {
    z2[j] = b2[j];
    for (size_t i = 0; i < L1_UNITS; i++) z2[j] += a1[i] * W2[i][j];
    a2[j] = relu(z2[j]);
  }
  // Capa 3
  for (size_t j = 0; j < L3_UNITS; j++) {
    z3[j] = b3[j];
    for (size_t i = 0; i < L2_UNITS; i++) z3[j] += a2[i] * W3[i][j];
    a3_val[j] = relu(z3[j]);
  }
  // Capa 4 (logits)
  for (size_t j = 0; j < OUTPUT_UNITS; j++) {
    z4[j] = b4[j];
    for (size_t i = 0; i < L3_UNITS; i++) z4[j] += a3_val[i] * W4[i][j];
  }

  // Softmax
  float maxZ = z4[0];
  for (size_t j = 1; j < OUTPUT_UNITS; j++)
    if (z4[j] > maxZ) maxZ = z4[j];

  float sumExp = 0;
  for (size_t j = 0; j < OUTPUT_UNITS; j++) {
    a4[j] = expf(z4[j] - maxZ);
    sumExp += a4[j];
  }
  for (size_t j = 0; j < OUTPUT_UNITS; j++)
    a4[j] /= sumExp;

  // argmax
  int predClass = 0;
  float maxProb = a4[0];
  for (size_t j = 1; j < OUTPUT_UNITS; j++) {
    if (a4[j] > maxProb) { maxProb = a4[j]; predClass = (int)j; }
  }
  *confidence = maxProb;
  return predClass;
}

// ==========================================
// BACKWARD PASS (SGD, cross-entropy 3-class)
// ==========================================
void trainLocalStep(const float raw_features[FEATURE_COUNT], int true_label) {
  float conf;
  predictLocal(raw_features, &conf);

  // dZ4 = a4 - one_hot(true_label)
  float dZ4[OUTPUT_UNITS];
  for (size_t j = 0; j < OUTPUT_UNITS; j++)
    dZ4[j] = a4[j] - ((int)j == true_label ? 1.0f : 0.0f);

  // dZ3
  float dZ3[L3_UNITS];
  for (size_t i = 0; i < L3_UNITS; i++) {
    float err = 0;
    for (size_t j = 0; j < OUTPUT_UNITS; j++) err += dZ4[j] * W4[i][j];
    dZ3[i] = err * relu_deriv(z3[i]);
  }
  // dZ2
  float dZ2[L2_UNITS];
  for (size_t i = 0; i < L2_UNITS; i++) {
    float err = 0;
    for (size_t j = 0; j < L3_UNITS; j++) err += dZ3[j] * W3[i][j];
    dZ2[i] = err * relu_deriv(z2[i]);
  }
  // dZ1
  float dZ1[L1_UNITS];
  for (size_t i = 0; i < L1_UNITS; i++) {
    float err = 0;
    for (size_t j = 0; j < L2_UNITS; j++) err += dZ2[j] * W2[i][j];
    dZ1[i] = err * relu_deriv(z1[i]);
  }

  // Actualizar W4, b4
  for (size_t j = 0; j < OUTPUT_UNITS; j++) {
    b4[j] -= LEARNING_RATE * dZ4[j];
    for (size_t i = 0; i < L3_UNITS; i++) W4[i][j] -= LEARNING_RATE * a3_val[i] * dZ4[j];
  }
  // Actualizar W3, b3
  for (size_t j = 0; j < L3_UNITS; j++) {
    b3[j] -= LEARNING_RATE * dZ3[j];
    for (size_t i = 0; i < L2_UNITS; i++) W3[i][j] -= LEARNING_RATE * a2[i] * dZ3[j];
  }
  // Actualizar W2, b2
  for (size_t j = 0; j < L2_UNITS; j++) {
    b2[j] -= LEARNING_RATE * dZ2[j];
    for (size_t i = 0; i < L1_UNITS; i++) W2[i][j] -= LEARNING_RATE * a1[i] * dZ2[j];
  }
  // Actualizar W1, b1
  for (size_t j = 0; j < L1_UNITS; j++) {
    b1[j] -= LEARNING_RATE * dZ1[j];
    for (size_t i = 0; i < FEATURE_COUNT; i++) W1[i][j] -= LEARNING_RATE * a0[i] * dZ1[j];
  }
  numSamples++;
}

// ==========================================
// IDS: ANALIZAR Y ALERTAR (3 clases)
// ==========================================
void analyzeAndAlert(float features[FEATURE_COUNT], const char* source, bool ruleTriggered) {
  float confidence;
  int predClass = predictLocal(features, &confidence);

  Serial.print("[IDS] "); Serial.print(source);
  Serial.print(" | pkts="); Serial.print((int)features[0]);
  for (size_t j = 0; j < OUTPUT_UNITS; j++) {
    Serial.print(" | "); Serial.print(CLASS_NAMES_STR[j]); Serial.print("=");
    Serial.print(a4[j] * 100, 1); Serial.print("%");
  }
  Serial.print(" -> "); Serial.println(CLASS_NAMES_STR[predClass]);

  if (predClass != 0) {
    totalAlertas++;
    setLED(predClass == 1 ? 255 : 0, 0, predClass == 2 ? 255 : 0);
    ledOffTime = millis() + 3000;

    Serial.println("========================================");
    Serial.print("  ATAQUE DETECTADO: "); Serial.println(CLASS_NAMES_STR[predClass]);
    Serial.print("  Confianza: "); Serial.print(confidence * 100, 1); Serial.println("%");
    Serial.println("========================================");

    StaticJsonDocument<512> alertDoc;
    alertDoc["alert"] = true;
    alertDoc["attack_type"] = CLASS_NAMES_STR[predClass];
    alertDoc["attack_probability"] = confidence;
    alertDoc["predicted_class"] = predClass;
    alertDoc["source"] = source;
    alertDoc["total_alerts"] = totalAlertas;
    char alertBuf[512]; size_t alertLen = serializeJson(alertDoc, alertBuf);
    myBroker.publish(TOPIC_ALERTS, alertBuf, alertLen, false);
  } else {
    setLED(0, 10, 0);
  }

  if ((uint32_t)features[0] >= MIN_PKTS_FOR_ML) {
    int hLabel = heuristicLabel(features, ruleTriggered);
    if (hLabel >= 0) {
      trainLocalStep(features, hLabel);
      Serial.print("[FL] Train label="); Serial.print(CLASS_NAMES_STR[hLabel]);
      Serial.print(" | Acum: "); Serial.println(numSamples);
    }
    if (numSamples >= SAMPLES_PER_UPDATE) {
      sendDeltasToRaspberry();
    }
  }
}

// ==========================================
// PROCESAR FEATURES DE CLIENTES
// ==========================================
void processReceivedFeatures(const char* payload, unsigned int length) {
  StaticJsonDocument<1024> doc;
  if (deserializeJson(doc, payload, length)) return;
  String senderID = doc["client_id"] | "unknown";
  JsonArray featArr = doc["features"].as<JsonArray>();
  if (featArr.size() != FEATURE_COUNT) return;
  float features[FEATURE_COUNT];
  for (size_t i = 0; i < FEATURE_COUNT; i++) features[i] = featArr[i].as<float>();
  Serial.print("[BROKER] Features de: "); Serial.println(senderID);
  uint32_t npkts = (uint32_t)features[0];
  bool rule = (npkts >= RULE_PKTS_ALERT);
  analyzeAndAlert(features, senderID.c_str(), rule);
}

// ==========================================
// ENVIAR DELTAS A RASPBERRY (3 clases)
// ==========================================
void sendDeltasToRaspberry() {
  if (!mqttRasp.connected()) {
    Serial.print("[FL] Conectando a Raspberry MQTT...");
    if (mqttRasp.connect(CLIENT_ID.c_str())) {
      Serial.println(" OK!");
      mqttRasp.subscribe(TOPIC_GLOBAL_MODEL);
    } else {
      Serial.println(" FALLO.");
      return;
    }
  }

  String json = "{";
  json += "\"client_id\":\"" + CLIENT_ID + "\",";
  json += "\"round\":" + String(currentRound) + ",";
  json += "\"num_samples\":" + String(numSamples) + ",";
  json += "\"model_arch\":\"13-32-16-8-3\",";
  json += "\"weight_delta\":{";

  // W4 [L3_UNITS=8][OUTPUT_UNITS=3]
  json += "\"W4\":[";
  for (size_t i = 0; i < L3_UNITS; i++) {
    json += "[";
    for (size_t j = 0; j < OUTPUT_UNITS; j++) {
      json += String(W4[i][j] - W4_base[i][j], 6);
      if (j < OUTPUT_UNITS - 1) json += ",";
    }
    json += "]";
    if (i < L3_UNITS - 1) json += ",";
  }
  json += "],";

  // b4 [OUTPUT_UNITS=3]
  json += "\"b4\":[";
  for (size_t j = 0; j < OUTPUT_UNITS; j++) {
    json += String(b4[j] - b4_base[j], 6);
    if (j < OUTPUT_UNITS - 1) json += ",";
  }
  json += "],";

  // W3 [L2_UNITS=16][L3_UNITS=8]
  json += "\"W3\":[";
  for (size_t i = 0; i < L2_UNITS; i++) {
    json += "[";
    for (size_t j = 0; j < L3_UNITS; j++) {
      json += String(W3[i][j] - W3_base[i][j], 6);
      if (j < L3_UNITS - 1) json += ",";
    }
    json += "]";
    if (i < L2_UNITS - 1) json += ",";
  }
  json += "],";

  // b3 [L3_UNITS=8]
  json += "\"b3\":[";
  for (size_t j = 0; j < L3_UNITS; j++) {
    json += String(b3[j] - b3_base[j], 6);
    if (j < L3_UNITS - 1) json += ",";
  }
  json += "]}}";

  bool ok = mqttRasp.publish(TOPIC_UPDATE, json.c_str());
  if (ok) {
    Serial.print("[FL] Deltas enviados. Ronda: "); Serial.println(currentRound);
    numSamples = 0;
    currentRound++;
  } else {
    Serial.println("[FL] Error enviando deltas.");
  }
}

// ==========================================
// CALLBACK MQTT EXTERNO: modelo global
// ==========================================
void onMqttRaspCallback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) == TOPIC_GLOBAL_MODEL) {
    Serial.println("\n========================================");
    Serial.println(" MODELO GLOBAL 3-CLASS RECIBIDO");
    Serial.println("========================================");

    DynamicJsonDocument doc(8192);
    DeserializationError err = deserializeJson(doc, payload, length);
    if (err) {
      Serial.print("[HFL] Error JSON: "); Serial.println(err.c_str());
      return;
    }

    int newRound = doc["round"] | 0;
    JsonArray w3arr = doc["W3"].as<JsonArray>();
    JsonArray b3arr = doc["b3"].as<JsonArray>();
    JsonArray w4arr = doc["W4"].as<JsonArray>();
    JsonArray b4arr = doc["b4"].as<JsonArray>();

    if (w3arr.size() != L2_UNITS || b3arr.size() != L3_UNITS) {
      Serial.println("[HFL] Error dimensiones W3/b3"); return;
    }

    // Sobreescribir W3 combinando _base con el delta global
    for (size_t i = 0; i < L2_UNITS; i++) {
      JsonArray row = w3arr[i].as<JsonArray>();
      if (row.size() != L3_UNITS) continue;
      for (size_t j = 0; j < L3_UNITS; j++) W3[i][j] = W3_base[i][j] + row[j].as<float>();
    }
    for (size_t j = 0; j < L3_UNITS; j++) b3[j] = b3_base[j] + b3arr[j].as<float>();

    // Sobreescribir W4 [8][3]
    if (w4arr.size() == L3_UNITS) {
      for (size_t i = 0; i < L3_UNITS; i++) {
        if (w4arr[i].is<JsonArray>()) {
          JsonArray row = w4arr[i].as<JsonArray>();
          for (size_t j = 0; j < OUTPUT_UNITS && j < row.size(); j++)
            W4[i][j] = W4_base[i][j] + row[j].as<float>();
        }
      }
    }
    // Sobreescribir b4 [3]
    if (b4arr.size() >= OUTPUT_UNITS) {
      for (size_t j = 0; j < OUTPUT_UNITS; j++) b4[j] = b4_base[j] + b4arr[j].as<float>();
    }

    currentRound = newRound;
    numSamples = 0;
    Serial.print("[HFL] Ronda: "); Serial.println(currentRound);
    Serial.println("[HFL] Pesos actualizados.\n");
    setLED(30, 30, 0);
    ledOffTime = millis() + 2000;
  }
}

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(2000);
  setLED(0, 10, 0);

  Serial.println("========================================");
  Serial.println(" ESP32-S3 BROKER + IDS 3-CLASS + HFL v6");
  Serial.println(" MLP: 13->32->16->8->3 (softmax)");
  Serial.println(" Clases: normal, mqtt_bruteforce, scan_A");
  Serial.println("========================================");

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("[BROKER] AP: "); Serial.println(AP_SSID);
  Serial.print("[BROKER] AP IP: "); Serial.println(WiFi.softAPIP());

  WiFi.begin(STA_SSID, STA_PASS);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(500); Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[BROKER] Red IP: "); Serial.println(WiFi.localIP());
  }

  myBroker.init(1883);
  mqttRasp.setServer(RASP_MQTT_SERVER, RASP_MQTT_PORT);
  mqttRasp.setBufferSize(8192);
  mqttRasp.setCallback(onMqttRaspCallback);

  initModel();
  resetBrokerFlow();
  brokerLastWindowMs = millis();
  Serial.println("[BROKER] Listo.\n");
}

// ==========================================
// LOOP
// ==========================================
void loop() {
  myBroker.update();

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttRasp.connected()) {
      if (mqttRasp.connect(CLIENT_ID.c_str())) {
        mqttRasp.subscribe(TOPIC_GLOBAL_MODEL);
        Serial.println("[MQTT] Conectado a Raspberry + suscrito a fl/global_model");
      }
    }
    mqttRasp.loop();
  }

  if (millis() - brokerLastWindowMs >= BROKER_WINDOW_MS) {
    brokerLastWindowMs = millis();
    if (brokerGlobalPkts > 0) {
      float features[FEATURE_COUNT];
      brokerExtractFeatures(features);
      Serial.print("\n[VENTANA] pkts="); Serial.print(brokerGlobalPkts);
      Serial.print(" conns="); Serial.print(brokerConnections);
      Serial.print(" bytes="); Serial.println(brokerGlobalBytes);

      bool ruleTriggered = (brokerGlobalPkts >= RULE_PKTS_ALERT) || (brokerConnections >= RULE_PKTS_ALERT);
      if (brokerGlobalPkts >= MIN_PKTS_FOR_ML || ruleTriggered) {
        analyzeAndAlert(features, "broker_direct", ruleTriggered);
      } else {
        Serial.print("[IDS] Solo "); Serial.print(brokerGlobalPkts);
        Serial.println(" pkts -> NORMAL");
        trainLocalStep(features, 0);
        Serial.print("[FL] Train label=normal | Acum: "); Serial.println(numSamples);
        if (numSamples >= SAMPLES_PER_UPDATE) sendDeltasToRaspberry();
      }
      resetBrokerFlow();
    }
  }

  if (ledOffTime > 0 && millis() > ledOffTime) {
    setLED(0, 10, 0);
    ledOffTime = 0;
  }
  delay(1);
}
