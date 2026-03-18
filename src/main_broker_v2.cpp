// =====================================================================
// main_broker.cpp - ESP32-S3 MQTT Broker + IDS + Federated Learning
// =====================================================================
// Modelo: MLP 13->32->16->8->1 (binario: normal/ataque)
// Features: 13 (flujo de red MQTT-IoT-IDS2020)
// Threshold: 0.95
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <sMQTTBroker.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>
#include "model_weights.h"

// LED RGB integrado (ESP32-S3 DevKit = GPIO 48)
#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48
#endif

// ==========================================
// CONFIGURACION WiFi (Access Point + Station)
// ==========================================
const char* AP_SSID = "FL_BROKER_NET";
const char* AP_PASS = "federated123";

const char* STA_SSID = "JAIMES_PUERTO 2.4";
const char* STA_PASS = "Anderson123";

const char* RASP_MQTT_SERVER = "192.168.1.21";
const int RASP_MQTT_PORT = 1883;

// ==========================================
// TOPICS MQTT
// ==========================================
const char* TOPIC_FEATURES = "fl/features";
const char* TOPIC_ALERTS   = "fl/alerts";
const char* TOPIC_UPDATE   = "fl/updates";

const String CLIENT_ID = "esp32_broker_01";

// ==========================================
// PARAMETROS DEL MODELO MLP (13->32->16->8->1)
// ==========================================
// constexpr size_t FEATURE_COUNT = 13; // NOTA: Ya viene exportada dentro de model_weights.h
constexpr size_t L1_UNITS = 32;
constexpr size_t L2_UNITS = 16;
constexpr size_t L3_UNITS = 8;
constexpr size_t OUTPUT_UNITS = 1;

const float ATTACK_THRESHOLD = 0.95f;
const float LEARNING_RATE = 0.001f;

int currentRound = 0;
int numSamples = 0;
constexpr int SAMPLES_PER_UPDATE = 5;

const char* CLASS_NAMES[2] = { "normal", "attack" };

static const char* FEATURE_NAMES[FEATURE_COUNT] = {
  "num_pkts", "mean_iat", "std_iat", "min_iat", "max_iat",
  "mean_pkt_len", "num_bytes", "num_psh_flags", "num_rst_flags",
  "num_urg_flags", "std_pkt_len", "min_pkt_len", "max_pkt_len"
};

// ==========================================
// PESOS DEL MODELO (copia mutable)
// ==========================================
float W1[FEATURE_COUNT][L1_UNITS];
float b1[L1_UNITS];
float W2[L1_UNITS][L2_UNITS];
float b2[L2_UNITS];
float W3[L2_UNITS][L3_UNITS];
float b3[L3_UNITS];
float W4[L3_UNITS][OUTPUT_UNITS];
float b4[OUTPUT_UNITS];

// Activaciones intermedias (para forward + backward)
float a0[FEATURE_COUNT];
float z1[L1_UNITS], a1[L1_UNITS];
float z2[L2_UNITS], a2[L2_UNITS];
float z3[L3_UNITS], a3[L3_UNITS];
float z4[OUTPUT_UNITS], a4[OUTPUT_UNITS];

// ==========================================
// OBJETOS MQTT
// ==========================================
WiFiClient raspClient;
PubSubClient mqttRasp(raspClient);

int totalAlertas = 0;
unsigned long ledOffTime = 0;

// Forward declarations
void processReceivedFeatures(const char* payload, unsigned int length);
void sendDeltasToRaspberry();

// ==========================================
// CLASE BROKER MQTT
// ==========================================
class MyBroker : public sMQTTBroker {
public:
  bool onEvent(sMQTTEvent *event) override {
    switch (event->Type()) {
      case NewClient_sMQTTEventType: {
        sMQTTNewClientEvent *e = (sMQTTNewClientEvent*)event;
        Serial.print("[BROKER] Cliente conectado: ");
        Serial.println(e->Client()->getClientId().c_str());
        break;
      }
      case LostConnect_sMQTTEventType: {
        Serial.println("[BROKER] Cliente desconectado.");
        break;
      }
      case Subscribe_sMQTTEventType: {
        sMQTTSubUnSubClientEvent *e = (sMQTTSubUnSubClientEvent*)event;
        Serial.print("[BROKER] Suscripcion a: ");
        Serial.println(e->Topic().c_str());
        break;
      }
      case Public_sMQTTEventType: {
        sMQTTPublicClientEvent *e = (sMQTTPublicClientEvent*)event;
        String topic = e->Topic().c_str();
        String payload = e->Payload().c_str();
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
// INICIALIZACION DEL MODELO
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

  Serial.println("[BROKER] Modelo MLP 13->32->16->8->1 inicializado.");
}

// ==========================================
// LED RGB
// ==========================================
void setLED(uint8_t r, uint8_t g, uint8_t b) {
  neopixelWrite(RGB_BUILTIN, r, g, b);
}

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
// FORWARD PASS (sigmoid output)
// ==========================================
float predictLocal(const float raw_features[FEATURE_COUNT]) {
  // Estandarizar con scaler (mean/std del entrenamiento)
  for (size_t i = 0; i < FEATURE_COUNT; i++) {
    a0[i] = (raw_features[i] - scaler_mean[i]) / scaler_std[i];
  }

  // Capa 1: Dense(32) + ReLU (BN folded en pesos)
  for (size_t j = 0; j < L1_UNITS; j++) {
    z1[j] = b1[j];
    for (size_t i = 0; i < FEATURE_COUNT; i++) z1[j] += a0[i] * W1[i][j];
    a1[j] = relu(z1[j]);
  }

  // Capa 2: Dense(16) + ReLU (BN folded en pesos)
  for (size_t j = 0; j < L2_UNITS; j++) {
    z2[j] = b2[j];
    for (size_t i = 0; i < L1_UNITS; i++) z2[j] += a1[i] * W2[i][j];
    a2[j] = relu(z2[j]);
  }

  // Capa 3: Dense(8) + ReLU
  for (size_t j = 0; j < L3_UNITS; j++) {
    z3[j] = b3[j];
    for (size_t i = 0; i < L2_UNITS; i++) z3[j] += a2[i] * W3[i][j];
    a3[j] = relu(z3[j]);
  }

  // Capa 4: Dense(1) + Sigmoid
  z4[0] = b4[0];
  for (size_t i = 0; i < L3_UNITS; i++) z4[0] += a3[i] * W4[i][0];
  a4[0] = sigmoid(z4[0]);

  return a4[0];  // probabilidad de ataque [0, 1]
}

// ==========================================
// BACKWARD PASS (Binary Cross-Entropy + SGD)
// ==========================================
void trainLocalStep(const float raw_features[FEATURE_COUNT], int true_label) {
  float prob = predictLocal(raw_features);

  // Gradiente de salida: dL/dz4 = a4 - y (BCE + sigmoid)
  float dZ4 = a4[0] - (float)true_label;

  // Gradientes capa 3
  float dZ3[L3_UNITS];
  for (size_t i = 0; i < L3_UNITS; i++) {
    dZ3[i] = dZ4 * W4[i][0] * relu_deriv(z3[i]);
  }

  // Gradientes capa 2
  float dZ2[L2_UNITS];
  for (size_t i = 0; i < L2_UNITS; i++) {
    float err = 0.0f;
    for (size_t j = 0; j < L3_UNITS; j++) err += dZ3[j] * W3[i][j];
    dZ2[i] = err * relu_deriv(z2[i]);
  }

  // Gradientes capa 1
  float dZ1[L1_UNITS];
  for (size_t i = 0; i < L1_UNITS; i++) {
    float err = 0.0f;
    for (size_t j = 0; j < L2_UNITS; j++) err += dZ2[j] * W2[i][j];
    dZ1[i] = err * relu_deriv(z1[i]);
  }

  // Actualizar pesos (SGD)
  // Capa 4
  b4[0] -= LEARNING_RATE * dZ4;
  for (size_t i = 0; i < L3_UNITS; i++) W4[i][0] -= LEARNING_RATE * a3[i] * dZ4;

  // Capa 3
  for (size_t j = 0; j < L3_UNITS; j++) {
    b3[j] -= LEARNING_RATE * dZ3[j];
    for (size_t i = 0; i < L2_UNITS; i++) W3[i][j] -= LEARNING_RATE * a2[i] * dZ3[j];
  }

  // Capa 2
  for (size_t j = 0; j < L2_UNITS; j++) {
    b2[j] -= LEARNING_RATE * dZ2[j];
    for (size_t i = 0; i < L1_UNITS; i++) W2[i][j] -= LEARNING_RATE * a1[i] * dZ2[j];
  }

  // Capa 1
  for (size_t j = 0; j < L1_UNITS; j++) {
    b1[j] -= LEARNING_RATE * dZ1[j];
    for (size_t i = 0; i < FEATURE_COUNT; i++) W1[i][j] -= LEARNING_RATE * a0[i] * dZ1[j];
  }

  numSamples++;
}

// ==========================================
// PROCESAR FEATURES RECIBIDAS
// ==========================================
void processReceivedFeatures(const char* payload, unsigned int length) {
  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload, length);

  if (err) {
    Serial.print("[BROKER] Error parseando JSON: ");
    Serial.println(err.c_str());
    return;
  }

  String senderID = doc["client_id"] | "unknown";
  JsonArray featArr = doc["features"].as<JsonArray>();

  if (featArr.size() != FEATURE_COUNT) {
    Serial.print("[BROKER] Error: se esperaban 13 features, llegaron ");
    Serial.println(featArr.size());
    return;
  }

  float features[FEATURE_COUNT];
  for (size_t i = 0; i < FEATURE_COUNT; i++) {
    features[i] = featArr[i].as<float>();
  }

  Serial.print("[BROKER] Features de "); Serial.println(senderID);

  // 1. INFERENCIA
  float attackProb = predictLocal(features);
  int pred = (attackProb >= ATTACK_THRESHOLD) ? 1 : 0;

  Serial.print("[BROKER] Probabilidad de ataque: ");
  Serial.print(attackProb * 100, 1);
  Serial.print("% -> ");
  Serial.println(CLASS_NAMES[pred]);

  // 2. ALERTA si es ataque
  if (pred == 1) {
    totalAlertas++;

    // LED AZUL por 3 segundos
    setLED(0, 0, 255);
    ledOffTime = millis() + 3000;

    Serial.println("========================================");
    Serial.println("  ATAQUE DETECTADO!");
    Serial.print("  Probabilidad: "); Serial.print(attackProb * 100, 1); Serial.println("%");
    Serial.print("  Fuente: "); Serial.println(senderID);
    Serial.println("========================================");

    StaticJsonDocument<512> alertDoc;
    alertDoc["alert"] = true;
    alertDoc["attack_type"] = "DoS";
    alertDoc["attack_probability"] = attackProb;
    alertDoc["source"] = senderID;
    alertDoc["total_alerts"] = totalAlertas;

    char alertBuf[512];
    size_t alertLen = serializeJson(alertDoc, alertBuf);
    myBroker.publish(TOPIC_ALERTS, alertBuf, alertLen, false);
    Serial.println("[BROKER] Alerta publicada en fl/alerts");
  } else {
    Serial.println("[BROKER] Trafico NORMAL.");
    setLED(0, 10, 0);  // verde tenue = normal
  }

  // 3. ENTRENAMIENTO LOCAL (usa la prediccion como label)
  trainLocalStep(features, pred);
  Serial.print("[BROKER] SGD completado. Muestras: ");
  Serial.println(numSamples);

  // 4. ENVIAR DELTAS cada N muestras
  if (numSamples >= SAMPLES_PER_UPDATE) {
    sendDeltasToRaspberry();
  }
}

// ==========================================
// ENVIAR DELTAS A RASPBERRY
// ==========================================
void sendDeltasToRaspberry() {
  if (!mqttRasp.connected()) {
    Serial.print("[BROKER] Conectando a Raspberry MQTT...");
    if (mqttRasp.connect(CLIENT_ID.c_str())) {
      Serial.println(" OK!");
    } else {
      Serial.println(" FALLO.");
      return;
    }
  }

  // Enviar deltas de la ultima capa (W4, b4) + penultima (W3, b3)
  String json = "{";
  json += "\"client_id\":\"" + CLIENT_ID + "\",";
  json += "\"round\":" + String(currentRound) + ",";
  json += "\"num_samples\":" + String(numSamples) + ",";
  json += "\"weight_delta\":{";

  // W4 deltas [L3_UNITS][1]
  json += "\"W4\":[";
  for (size_t i = 0; i < L3_UNITS; i++) {
    float delta = W4[i][0] - W4_base[i][0];
    json += String(delta, 6);
    if (i < L3_UNITS - 1) json += ",";
  }
  json += "],";

  // b4 deltas [1]
  json += "\"b4\":[";
  json += String(b4[0] - b4_base[0], 6);
  json += "],";

  // W3 deltas [L2_UNITS][L3_UNITS]
  json += "\"W3\":[";
  for (size_t i = 0; i < L2_UNITS; i++) {
    json += "[";
    for (size_t j = 0; j < L3_UNITS; j++) {
      float delta = W3[i][j] - W3_base[i][j];
      json += String(delta, 6);
      if (j < L3_UNITS - 1) json += ",";
    }
    json += "]";
    if (i < L2_UNITS - 1) json += ",";
  }
  json += "],";

  // b3 deltas [L3_UNITS]
  json += "\"b3\":[";
  for (size_t j = 0; j < L3_UNITS; j++) {
    float delta = b3[j] - b3_base[j];
    json += String(delta, 6);
    if (j < L3_UNITS - 1) json += ",";
  }
  json += "]}}";

  bool ok = mqttRasp.publish(TOPIC_UPDATE, json.c_str());
  if (ok) {
    Serial.println("[BROKER] Deltas enviados a Raspberry via MQTT.");
    numSamples = 0;
    currentRound++;
  } else {
    Serial.println("[BROKER] Error enviando deltas.");
  }
}

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(2000);

  // LED verde al arrancar
  setLED(0, 10, 0);

  Serial.println("========================================");
  Serial.println(" ESP32-S3 BROKER MQTT + IDS (binario)");
  Serial.println(" Modelo: 13->32->16->8->1 sigmoid");
  Serial.println(" Threshold: 0.95");
  Serial.println("========================================");

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("[BROKER] AP: "); Serial.println(AP_SSID);
  Serial.print("[BROKER] AP IP: "); Serial.println(WiFi.softAPIP());

  WiFi.begin(STA_SSID, STA_PASS);
  Serial.print("[BROKER] Conectando a red externa");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(500); Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[BROKER] Red externa IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("[BROKER] AVISO: Sin red externa.");
  }

  myBroker.init(1883);
  Serial.println("[BROKER] Broker MQTT en puerto 1883");

  mqttRasp.setServer(RASP_MQTT_SERVER, RASP_MQTT_PORT);
  mqttRasp.setBufferSize(4096);

  initModel();
  Serial.println("[BROKER] Esperando clientes ESP32...\n");
}

// ==========================================
// LOOP
// ==========================================
void loop() {
  myBroker.update();
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttRasp.connected()) mqttRasp.connect(CLIENT_ID.c_str());
    mqttRasp.loop();
  }

  // Apagar LED azul despues de 3s
  if (ledOffTime > 0 && millis() > ledOffTime) {
    setLED(0, 10, 0);  // volver a verde
    ledOffTime = 0;
  }

  delay(1);
}