// =====================================================================
// main_broker_hfl.cpp — ESP32-S3 Broker + IDS + HFL Bidireccional
// =====================================================================
// CAMBIO CLAVE vs v4: Se suscribe a fl/global_model en el MQTT externo
// (Raspberry Pi) para recibir el modelo global del coordinador PC.
// Al recibirlo, sobreescribe los pesos locales W3, b3, W4, b4.
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
// CONFIGURACION WiFi
// ==========================================
const char* AP_SSID = "FL_BROKER_NET";
const char* AP_PASS = "federated123";
const char* STA_SSID = "TP-Link_AADB";       // <-- CAMBIAR
const char* STA_PASS = "55707954";            // <-- CAMBIAR
const char* RASP_MQTT_SERVER = "192.168.1.21"; // <-- IP Raspberry Pi
const int RASP_MQTT_PORT = 1883;

// ==========================================
// TOPICS MQTT
// ==========================================
const char* TOPIC_FEATURES     = "fl/features";
const char* TOPIC_ALERTS       = "fl/alerts";
const char* TOPIC_UPDATE       = "fl/updates";
const char* TOPIC_GLOBAL_MODEL = "fl/global_model"; // NUEVO: modelo del PC

const String CLIENT_ID = "esp32_broker_01";

// ==========================================
// MODELO MLP: 13 -> 32 -> 16 -> 8 -> 1
// ==========================================
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

// Pesos mutables (se sobreescriben con el modelo global)
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
float z4[OUTPUT_UNITS], a4[OUTPUT_UNITS];

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

// MQTT externo (hacia Raspberry Pi)
WiFiClient raspClient;
PubSubClient mqttRasp(raspClient);

int totalAlertas = 0;
unsigned long ledOffTime = 0;
bool modelReceivedFromPC = false;

// Forward declarations
void processReceivedFeatures(const char* payload, unsigned int length);
void sendDeltasToRaspberry();
void analyzeAndAlert(float features[FEATURE_COUNT], const char* source, bool ruleTriggered);
void onMqttRaspCallback(char* topic, byte* payload, unsigned int length);

// ==========================================
// HEURISTIC LABELING
// ==========================================
int heuristicLabel(float features[FEATURE_COUNT], bool ruleTriggered) {
    uint32_t pkts    = (uint32_t)features[0];
    float    meanIat = features[1];
    if (ruleTriggered) return 1;
    if (pkts >= 100) return 1;
    if (pkts <= 30 && meanIat >= 0.1f) return 0;
    if (meanIat > 0 && meanIat <= 0.01f && pkts > MIN_PKTS_FOR_ML) return 1;
    return -1;
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
// INICIALIZACION DEL MODELO (desde pesos base de model_weights.h)
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
  Serial.println("[MODEL] Pesos base cargados (model_weights.h)");
}

// ==========================================
// LED
// ==========================================
void setLED(uint8_t r, uint8_t g, uint8_t b) { neopixelWrite(RGB_BUILTIN, r, g, b); }

// ==========================================
// FUNCIONES MATEMATICAS
// ==========================================
float relu(float x) { return x > 0 ? x : 0.0f; }
float relu_deriv(float x) { return x > 0 ? 1.0f : 0.0f; }
float sigmoid(float x) {
  if (x > 20.0f) return 1.0f;
  if (x < -20.0f) return 0.0f;
  return 1.0f / (1.0f + expf(-x));
}

// ==========================================
// FORWARD PASS
// ==========================================
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

// ==========================================
// BACKWARD PASS (SGD)
// ==========================================
void trainLocalStep(const float raw_features[FEATURE_COUNT], int true_label) {
  float prob = predictLocal(raw_features);
  float dZ4 = a4[0] - (float)true_label;
  float dZ3[L3_UNITS];
  for (size_t i = 0; i < L3_UNITS; i++)
    dZ3[i] = dZ4 * W4[i][0] * relu_deriv(z3[i]);
  float dZ2[L2_UNITS];
  for (size_t i = 0; i < L2_UNITS; i++) {
    float err = 0; for (size_t j = 0; j < L3_UNITS; j++) err += dZ3[j] * W3[i][j];
    dZ2[i] = err * relu_deriv(z2[i]);
  }
  float dZ1[L1_UNITS];
  for (size_t i = 0; i < L1_UNITS; i++) {
    float err = 0; for (size_t j = 0; j < L2_UNITS; j++) err += dZ2[j] * W2[i][j];
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

// ==========================================
// IDS: ANALIZAR Y ALERTAR
// ==========================================
void analyzeAndAlert(float features[FEATURE_COUNT], const char* source, bool ruleTriggered) {
  float attackProb = predictLocal(features);
  int mlPred = (attackProb >= ATTACK_THRESHOLD) ? 1 : 0;
  int finalPred = (mlPred == 1 || ruleTriggered) ? 1 : 0;

  Serial.print("[IDS] "); Serial.print(source);
  Serial.print(" | pkts="); Serial.print((int)features[0]);
  Serial.print(" | ML="); Serial.print(attackProb * 100, 1); Serial.print("%");
  if (ruleTriggered) Serial.print(" | REGLA");
  Serial.print(" -> "); Serial.println(CLASS_NAMES[finalPred]);

  if (finalPred == 1) {
    totalAlertas++;
    setLED(0, 0, 255);
    ledOffTime = millis() + 3000;
    Serial.println("========================================");
    Serial.println("  ATAQUE DETECTADO!");
    Serial.print("  Prob: "); Serial.print(attackProb * 100, 1); Serial.println("%");
    Serial.println("========================================");

    StaticJsonDocument<512> alertDoc;
    alertDoc["alert"] = true;
    alertDoc["attack_type"] = "DoS";
    alertDoc["attack_probability"] = attackProb;
    alertDoc["rule_triggered"] = ruleTriggered;
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
      Serial.print("[FL] Train label="); Serial.print(hLabel);
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
// ENVIAR DELTAS A RASPBERRY PI (Bottom-Up)
// ==========================================
void sendDeltasToRaspberry() {
  if (!mqttRasp.connected()) {
    if (!mqttRasp.connect(CLIENT_ID.c_str())) return;
    mqttRasp.subscribe(TOPIC_GLOBAL_MODEL); // Re-suscribirse
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

// ================================================================
// NUEVO: Callback MQTT externo — recibir modelo global del PC
// ================================================================
void onMqttRaspCallback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) == TOPIC_GLOBAL_MODEL) {
    Serial.println("\n========================================");
    Serial.println(" MODELO GLOBAL RECIBIDO DEL COORDINADOR");
    Serial.println("========================================");

    // Parsear JSON con el modelo global
    // El payload puede ser grande (~2KB), usar DynamicJsonDocument
    DynamicJsonDocument doc(4096);
    DeserializationError err = deserializeJson(doc, payload, length);
    if (err) {
      Serial.print("[HFL] Error parseando modelo global: ");
      Serial.println(err.c_str());
      return;
    }

    int newRound = doc["round"] | 0;
    JsonArray w3arr = doc["W3"].as<JsonArray>();
    JsonArray b3arr = doc["b3"].as<JsonArray>();
    JsonArray w4arr = doc["W4"].as<JsonArray>();
    JsonArray b4arr = doc["b4"].as<JsonArray>();

    // Validar dimensiones
    if (w3arr.size() != L2_UNITS || b3arr.size() != L3_UNITS) {
      Serial.println("[HFL] Error: dimensiones W3/b3 incorrectas");
      return;
    }

    // Sobreescribir pesos W3
    for (size_t i = 0; i < L2_UNITS; i++) {
      JsonArray row = w3arr[i].as<JsonArray>();
      if (row.size() != L3_UNITS) continue;
      for (size_t j = 0; j < L3_UNITS; j++) {
        W3[i][j] = row[j].as<float>();
      }
    }
    // Sobreescribir b3
    for (size_t j = 0; j < L3_UNITS; j++) {
      b3[j] = b3arr[j].as<float>();
    }

    // Sobreescribir W4
    if (w4arr.size() == L3_UNITS) {
      // Puede venir como array de arrays [[v],[v],...] o como array plano [v,v,...]
      for (size_t i = 0; i < L3_UNITS; i++) {
        if (w4arr[i].is<JsonArray>()) {
          W4[i][0] = w4arr[i][0].as<float>();
        } else {
          W4[i][0] = w4arr[i].as<float>();
        }
      }
    }
    // Sobreescribir b4
    if (b4arr.size() >= 1) {
      b4[0] = b4arr[0].as<float>();
    }

    currentRound = newRound;
    numSamples = 0;
    modelReceivedFromPC = true;

    Serial.print("[HFL] Ronda: "); Serial.println(currentRound);
    Serial.println("[HFL] Pesos W3, b3, W4, b4 actualizados.");
    Serial.println("[HFL] Entrenamiento local reiniciado.\n");

    // Flash LED amarillo para indicar modelo recibido
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
  Serial.println(" ESP32-S3 BROKER + IDS + HFL");
  Serial.println(" Bidireccional: recibe modelo del PC");
  Serial.println(" MLP: 13->32->16->8->1 (th=0.95)");
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

  // MQTT externo (hacia Raspberry Pi)
  mqttRasp.setServer(RASP_MQTT_SERVER, RASP_MQTT_PORT);
  mqttRasp.setBufferSize(8192);
  mqttRasp.setCallback(onMqttRaspCallback);  // NUEVO: callback para modelo global

  initModel();
  resetBrokerFlow();
  brokerLastWindowMs = millis();
  Serial.println("[BROKER] Listo. Esperando modelo global del PC...\n");
}

// ==========================================
// LOOP
// ==========================================
void loop() {
  myBroker.update();

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttRasp.connected()) {
      if (mqttRasp.connect(CLIENT_ID.c_str())) {
        mqttRasp.subscribe(TOPIC_GLOBAL_MODEL);  // NUEVO: suscribirse al modelo global
        Serial.println("[MQTT] Conectado a Raspberry Pi + suscrito a fl/global_model");
      }
    }
    mqttRasp.loop();
  }

  // Ventana de monitoring del broker
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
        Serial.print("[IDS] Solo "); Serial.print(brokerGlobalPkts);
        Serial.println(" pkts -> NORMAL");
        trainLocalStep(features, 0);
        Serial.print("[FL] Train label=0 | Acum: "); Serial.println(numSamples);
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
