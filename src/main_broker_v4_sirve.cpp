// =====================================================================
// main_broker.cpp - ESP32-S3 MQTT Broker + IDS + FL
// FIX: etiqueta heuristica + ventanas normales entrenan como label=0
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

const char* AP_SSID = "FL_BROKER_NET";
const char* AP_PASS = "federated123";
const char* STA_SSID = "TP-Link_AADB";
const char* STA_PASS = "55707954";
const char* RASP_MQTT_SERVER = "192.168.1.21";
const int RASP_MQTT_PORT = 1883;

const char* TOPIC_FEATURES = "fl/features";
const char* TOPIC_ALERTS   = "fl/alerts";
const char* TOPIC_UPDATE   = "fl/updates";
const String CLIENT_ID = "esp32_broker_01";

constexpr size_t L1_UNITS = 32;
constexpr size_t L2_UNITS = 16;
constexpr size_t L3_UNITS = 8;
constexpr size_t OUTPUT_UNITS = 1;

const float ATTACK_THRESHOLD = 0.95f;
const float LEARNING_RATE = 0.001f;

constexpr uint32_t MIN_PKTS_FOR_ML = 10;
constexpr uint32_t RULE_PKTS_ALERT = 200;

int currentRound = 0;
int numSamples = 0;
constexpr int SAMPLES_PER_UPDATE = 20;

const char* CLASS_NAMES[2] = { "normal", "attack" };

float W1[FEATURE_COUNT][L1_UNITS];
float b1[L1_UNITS];
float W2[L1_UNITS][L2_UNITS];
float b2[L2_UNITS];
float W3[L2_UNITS][L3_UNITS];
float b3[L3_UNITS];
float W4[L3_UNITS][OUTPUT_UNITS];
float b4[OUTPUT_UNITS];

float a0[FEATURE_COUNT];
float z1[L1_UNITS], a1[L1_UNITS];
float z2[L2_UNITS], a2[L2_UNITS];
float z3[L3_UNITS], a3_val[L3_UNITS];
float z4[OUTPUT_UNITS], a4[OUTPUT_UNITS];

constexpr uint32_t BROKER_WINDOW_MS = 5000;

uint32_t brokerGlobalPkts = 0;
uint32_t brokerGlobalBytes = 0;
uint32_t brokerConnections = 0;
unsigned long brokerLastWindowMs = 0;
unsigned long brokerFirstPktUs = 0;
unsigned long brokerLastPktUs = 0;
float brokerSumIat = 0;
float brokerSumSqIat = 0;
float brokerMinIat = 1e9f;
float brokerMaxIat = 0;
float brokerSumPktLen = 0;
float brokerSumSqPktLen = 0;
float brokerMinPktLen = 1e9f;
float brokerMaxPktLen = 0;

WiFiClient raspClient;
PubSubClient mqttRasp(raspClient);

int totalAlertas = 0;
unsigned long ledOffTime = 0;

void processReceivedFeatures(const char* payload, unsigned int length);
void sendDeltasToRaspberry();
void analyzeAndAlert(float features[FEATURE_COUNT], const char* source, bool ruleTriggered);

// =======================================================================
//  CAMBIO 1: Etiqueta heuristica para entrenamiento
//  Retorna: 0=normal, 1=ataque, -1=incierto (no entrenar con esta muestra)
// =======================================================================
int heuristicLabel(float features[FEATURE_COUNT], bool ruleTriggered) {
    uint32_t pkts    = (uint32_t)features[0];
    float    meanIat = features[1];

    if (ruleTriggered) return 1;
    if (pkts >= 100) return 1;
    if (pkts <= 30 && meanIat >= 0.1f) return 0;
    if (meanIat > 0 && meanIat <= 0.01f && pkts > MIN_PKTS_FOR_ML) return 1;

    return -1;
}

void resetBrokerFlow() {
  brokerGlobalPkts = 0;
  brokerGlobalBytes = 0;
  brokerConnections = 0;
  brokerFirstPktUs = 0;
  brokerLastPktUs = 0;
  brokerSumIat = 0;
  brokerSumSqIat = 0;
  brokerMinIat = 1e9f;
  brokerMaxIat = 0;
  brokerSumPktLen = 0;
  brokerSumSqPktLen = 0;
  brokerMinPktLen = 1e9f;
  brokerMaxPktLen = 0;
}

void brokerTrackEvent(uint16_t pkt_len) {
  unsigned long now = micros();

  if (brokerGlobalPkts > 0 && brokerLastPktUs > 0) {
    float iat = (now - brokerLastPktUs) / 1e6f;
    brokerSumIat    += iat;
    brokerSumSqIat  += iat * iat;
    if (iat < brokerMinIat) brokerMinIat = iat;
    if (iat > brokerMaxIat) brokerMaxIat = iat;
  } else {
    brokerFirstPktUs = now;
  }

  brokerGlobalPkts++;
  brokerGlobalBytes += pkt_len;
  brokerSumPktLen    += (float)pkt_len;
  brokerSumSqPktLen  += (float)pkt_len * pkt_len;
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
  if (var_pkt < 0) var_pkt = 0;
  if (var_iat < 0) var_iat = 0;

  out[0]  = n;
  out[1]  = mean_iat;
  out[2]  = sqrtf(var_iat);
  out[3]  = (n > 1) ? brokerMinIat : 0.0f;
  out[4]  = (n > 1) ? brokerMaxIat : 0.0f;
  out[5]  = mean_pkt;
  out[6]  = (float)brokerGlobalBytes;
  out[7]  = 0.0f;
  out[8]  = 0.0f;
  out[9]  = 0.0f;
  out[10] = sqrtf(var_pkt);
  out[11] = (brokerMinPktLen < 1e8f) ? brokerMinPktLen : 0.0f;
  out[12] = brokerMaxPktLen;
}

class MyBroker : public sMQTTBroker {
public:
  bool onEvent(sMQTTEvent *event) override {
    switch (event->Type()) {
      case NewClient_sMQTTEventType: {
        brokerConnections++;
        brokerTrackEvent(64);
        if (brokerConnections <= 3 || brokerConnections % 50 == 0) {
          sMQTTNewClientEvent *e = (sMQTTNewClientEvent*)event;
          Serial.print("[BROKER] Cliente #");
          Serial.print(brokerConnections);
          Serial.print(": ");
          Serial.println(e->Client()->getClientId().c_str());
        }
        break;
      }
      case LostConnect_sMQTTEventType: {
        brokerTrackEvent(32);
        break;
      }
      case Subscribe_sMQTTEventType: {
        brokerTrackEvent(48);
        break;
      }
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

  Serial.println("[BROKER] Modelo MLP 13->32->16->8->1 inicializado.");
}

void setLED(uint8_t r, uint8_t g, uint8_t b) {
  neopixelWrite(RGB_BUILTIN, r, g, b);
}

float relu(float x) { return x > 0 ? x : 0.0f; }
float relu_deriv(float x) { return x > 0 ? 1.0f : 0.0f; }
float sigmoid(float x) {
  if (x > 20.0f) return 1.0f;
  if (x < -20.0f) return 0.0f;
  return 1.0f / (1.0f + expf(-x));
}

float predictLocal(const float raw_features[FEATURE_COUNT]) {
  for (size_t i = 0; i < FEATURE_COUNT; i++)
    a0[i] = (raw_features[i] - scaler_mean[i]) / scaler_std[i];

  for (size_t j = 0; j < L1_UNITS; j++) {
    z1[j] = b1[j];
    for (size_t i = 0; i < FEATURE_COUNT; i++) z1[j] += a0[i] * W1[i][j];
    a1[j] = relu(z1[j]);
  }
  for (size_t j = 0; j < L2_UNITS; j++) {
    z2[j] = b2[j];
    for (size_t i = 0; i < L1_UNITS; i++) z2[j] += a1[i] * W2[i][j];
    a2[j] = relu(z2[j]);
  }
  for (size_t j = 0; j < L3_UNITS; j++) {
    z3[j] = b3[j];
    for (size_t i = 0; i < L2_UNITS; i++) z3[j] += a2[i] * W3[i][j];
    a3_val[j] = relu(z3[j]);
  }
  z4[0] = b4[0];
  for (size_t i = 0; i < L3_UNITS; i++) z4[0] += a3_val[i] * W4[i][0];
  a4[0] = sigmoid(z4[0]);
  return a4[0];
}

void trainLocalStep(const float raw_features[FEATURE_COUNT], int true_label) {
  float prob = predictLocal(raw_features);
  float dZ4 = a4[0] - (float)true_label;

  float dZ3[L3_UNITS];
  for (size_t i = 0; i < L3_UNITS; i++)
    dZ3[i] = dZ4 * W4[i][0] * relu_deriv(z3[i]);

  float dZ2[L2_UNITS];
  for (size_t i = 0; i < L2_UNITS; i++) {
    float err = 0.0f;
    for (size_t j = 0; j < L3_UNITS; j++) err += dZ3[j] * W3[i][j];
    dZ2[i] = err * relu_deriv(z2[i]);
  }

  float dZ1[L1_UNITS];
  for (size_t i = 0; i < L1_UNITS; i++) {
    float err = 0.0f;
    for (size_t j = 0; j < L2_UNITS; j++) err += dZ2[j] * W2[i][j];
    dZ1[i] = err * relu_deriv(z1[i]);
  }

  b4[0] -= LEARNING_RATE * dZ4;
  for (size_t i = 0; i < L3_UNITS; i++) W4[i][0] -= LEARNING_RATE * a3_val[i] * dZ4;
  for (size_t j = 0; j < L3_UNITS; j++) {
    b3[j] -= LEARNING_RATE * dZ3[j];
    for (size_t i = 0; i < L2_UNITS; i++) W3[i][j] -= LEARNING_RATE * a2[i] * dZ3[j];
  }
  for (size_t j = 0; j < L2_UNITS; j++) {
    b2[j] -= LEARNING_RATE * dZ2[j];
    for (size_t i = 0; i < L1_UNITS; i++) W2[i][j] -= LEARNING_RATE * a1[i] * dZ2[j];
  }
  for (size_t j = 0; j < L1_UNITS; j++) {
    b1[j] -= LEARNING_RATE * dZ1[j];
    for (size_t i = 0; i < FEATURE_COUNT; i++) W1[i][j] -= LEARNING_RATE * a0[i] * dZ1[j];
  }
  numSamples++;
}

void analyzeAndAlert(float features[FEATURE_COUNT], const char* source, bool ruleTriggered) {
  float attackProb = predictLocal(features);
  int mlPred = (attackProb >= ATTACK_THRESHOLD) ? 1 : 0;
  int finalPred = (mlPred == 1 || ruleTriggered) ? 1 : 0;

  Serial.print("[IDS] ");
  Serial.print(source);
  Serial.print(" | pkts="); Serial.print((int)features[0]);
  Serial.print(" | conns="); Serial.print(brokerConnections);
  Serial.print(" | ML="); Serial.print(attackProb * 100, 1); Serial.print("%");
  if (ruleTriggered) Serial.print(" | REGLA: pkts>" + String(RULE_PKTS_ALERT));
  Serial.print(" -> "); Serial.println(CLASS_NAMES[finalPred]);

  if (finalPred == 1) {
    totalAlertas++;
    setLED(0, 0, 255);
    ledOffTime = millis() + 3000;

    Serial.println("========================================");
    Serial.println("  ATAQUE DETECTADO!");
    Serial.print("  Prob ML: "); Serial.print(attackProb * 100, 1); Serial.println("%");
    Serial.print("  Regla activada: "); Serial.println(ruleTriggered ? "SI" : "NO");
    Serial.print("  Fuente: "); Serial.println(source);
    Serial.print("  Pkts ventana: "); Serial.println((int)features[0]);
    Serial.print("  Conns ventana: "); Serial.println(brokerConnections);
    Serial.println("========================================");

    StaticJsonDocument<512> alertDoc;
    alertDoc["alert"] = true;
    alertDoc["attack_type"] = "DoS";
    alertDoc["attack_probability"] = attackProb;
    alertDoc["rule_triggered"] = ruleTriggered;
    alertDoc["source"] = source;
    alertDoc["total_alerts"] = totalAlertas;
    alertDoc["pkts_in_window"] = (int)features[0];
    alertDoc["conns_in_window"] = brokerConnections;

    char alertBuf[512];
    size_t alertLen = serializeJson(alertDoc, alertBuf);
    myBroker.publish(TOPIC_ALERTS, alertBuf, alertLen, false);
  } else {
    setLED(0, 10, 0);
  }

  // =======================================================================
  //  CAMBIO 1: Entrenar con etiqueta HEURISTICA, no con finalPred
  //  Antes: trainLocalStep(features, finalPred)  ← feedback loop
  //  Ahora: etiqueta basada en reglas del trafico, independiente del modelo
  // =======================================================================
  if ((uint32_t)features[0] >= MIN_PKTS_FOR_ML) {
    int hLabel = heuristicLabel(features, ruleTriggered);
    if (hLabel >= 0) {
      trainLocalStep(features, hLabel);
      Serial.print("[FL] Entrenado con label="); Serial.print(hLabel);
      Serial.print(" ("); Serial.print(hLabel == 0 ? "normal" : "ataque");
      Serial.print(") | Acumulado: "); Serial.println(numSamples);
    }
    if (numSamples >= SAMPLES_PER_UPDATE) {
      sendDeltasToRaspberry();
    }
  }
}

void processReceivedFeatures(const char* payload, unsigned int length) {
  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) return;

  String senderID = doc["client_id"] | "unknown";
  JsonArray featArr = doc["features"].as<JsonArray>();
  if (featArr.size() != FEATURE_COUNT) return;

  float features[FEATURE_COUNT];
  for (size_t i = 0; i < FEATURE_COUNT; i++)
    features[i] = featArr[i].as<float>();

  Serial.print("[BROKER] Features de: "); Serial.println(senderID);

  uint32_t npkts = (uint32_t)features[0];
  bool rule = (npkts >= RULE_PKTS_ALERT);
  analyzeAndAlert(features, senderID.c_str(), rule);
}

void sendDeltasToRaspberry() {
  if (!mqttRasp.connected()) {
    if (!mqttRasp.connect(CLIENT_ID.c_str())) return;
  }

  String json = "{";
  json += "\"client_id\":\"" + CLIENT_ID + "\",";
  json += "\"round\":" + String(currentRound) + ",";
  json += "\"num_samples\":" + String(numSamples) + ",";
  json += "\"model_arch\":\"13-32-16-8-1\",";
  json += "\"weight_delta\":{";

  json += "\"W4\":[";
  for (size_t i = 0; i < L3_UNITS; i++) {
    json += String(W4[i][0] - W4_base[i][0], 6);
    if (i < L3_UNITS - 1) json += ",";
  }
  json += "],";

  json += "\"b4\":[" + String(b4[0] - b4_base[0], 6) + "],";

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
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  setLED(0, 10, 0);

  Serial.println("========================================");
  Serial.println(" ESP32-S3 BROKER + IDS HIBRIDO");
  Serial.println(" ML: 13->32->16->8->1 (th=0.95)");
  Serial.println(" Regla: >" + String(RULE_PKTS_ALERT) + " pkts/ventana");
  Serial.println(" Min pkts ML: " + String(MIN_PKTS_FOR_ML));
  Serial.println(" FL: Etiqueta heuristica + ventanas normales");
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

  initModel();
  resetBrokerFlow();
  brokerLastWindowMs = millis();
  Serial.println("[BROKER] Listo.\n");
}

void loop() {
  myBroker.update();

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttRasp.connected()) mqttRasp.connect(CLIENT_ID.c_str());
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

      bool ruleTriggered = (brokerGlobalPkts >= RULE_PKTS_ALERT) ||
                           (brokerConnections >= RULE_PKTS_ALERT);

      if (brokerGlobalPkts >= MIN_PKTS_FOR_ML || ruleTriggered) {
        analyzeAndAlert(features, "broker_direct", ruleTriggered);
      } else {
        // =======================================================================
        //  CAMBIO 2: Ventanas con pocos paquetes = trafico normal
        //  Entrenar como label=0 para balancear las muestras de ataque
        // =======================================================================
        Serial.print("[IDS] Solo "); Serial.print(brokerGlobalPkts);
        Serial.print(" pkts (<"); Serial.print(MIN_PKTS_FOR_ML);
        Serial.println(") -> NORMAL");

        trainLocalStep(features, 0);
        Serial.print("[FL] Entrenado con label=0 (normal) | Acumulado: ");
        Serial.println(numSamples);

        if (numSamples >= SAMPLES_PER_UPDATE) {
          sendDeltasToRaspberry();
        }
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