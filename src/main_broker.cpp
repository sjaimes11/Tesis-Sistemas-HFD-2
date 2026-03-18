// =====================================================================
// main_broker.cpp - ESP32-S3 MQTT Broker + IDS + Federated Learning
// =====================================================================
// Modelo: MLP 36->64->32->6 (6 clases de ataque)
// Features: 36 (datos crudos, sin log1p ni clipping)
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <sMQTTBroker.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>
#include "model_weights.h"

// ==========================================
// CONFIGURACION WiFi (Access Point + Station)
// ==========================================
const char* AP_SSID = "FL_BROKER_NET";
const char* AP_PASS = "federated123";

// Red WiFi hacia la Raspberry Pi
const char* STA_SSID = "JAIMES_PUERTO 2.4";
const char* STA_PASS = "Anderson123";

// IP de la Raspberry Pi (MQTT externo)
const char* RASP_MQTT_SERVER = "192.168.40.57";
const int RASP_MQTT_PORT = 1883;

// ==========================================
// TOPICS MQTT
// ==========================================
const char* TOPIC_FEATURES = "fl/features";
const char* TOPIC_ALERTS   = "fl/alerts";
const char* TOPIC_UPDATE   = "fl/updates";

const String CLIENT_ID = "esp32_broker_01";

// ==========================================
// PARAMETROS DEL MODELO MLP (36->64->32->6)
// ==========================================
constexpr size_t FEATURE_COUNT = 36;
constexpr size_t L1_UNITS = 64;
constexpr size_t L2_UNITS = 32;
constexpr size_t NUM_CLASSES = 6;
const float LEARNING_RATE = 0.001f;

int currentRound = 0;
int numSamples = 0;
constexpr int SAMPLES_PER_UPDATE = 5;

// 6 clases del label_map.json
const char* CLASS_NAMES[NUM_CLASSES] = {
  "benign",
  "ddos_ack_fragmentation",
  "ddos_icmp_flood",
  "ddos_tcp_flood",
  "dos_syn_flood",
  "dos_tcp_flood"
};

// ==========================================
// PESOS DEL MODELO (Copia mutable)
// ==========================================
float W1[FEATURE_COUNT][L1_UNITS];
float b1[L1_UNITS];
float W2[L1_UNITS][L2_UNITS];
float b2[L2_UNITS];
float W3[L2_UNITS][NUM_CLASSES];
float b3[NUM_CLASSES];

// Activaciones intermedias
float a0[FEATURE_COUNT];
float z1[L1_UNITS], a1[L1_UNITS];
float z2[L2_UNITS], a2[L2_UNITS];
float z3[NUM_CLASSES], a3[NUM_CLASSES];

// ==========================================
// OBJETOS MQTT
// ==========================================
WiFiClient raspClient;
PubSubClient mqttRasp(raspClient);

int totalAlertas = 0;

// Forward declarations
void processReceivedFeatures(const char* payload, unsigned int length);
void sendDeltasToRaspberry();

// ==========================================
// CLASE BROKER MQTT (debe ir antes de processReceivedFeatures)
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
    for (size_t j = 0; j < NUM_CLASSES; j++) W3[i][j] = W3_base[i][j];
  for (size_t j = 0; j < NUM_CLASSES; j++) b3[j] = b3_base[j];

  Serial.println("[BROKER] Modelo MLP 36->64->32->6 inicializado.");
}

// ==========================================
// FUNCIONES MATEMATICAS
// ==========================================
float relu(float x) { return x > 0 ? x : 0.0f; }
float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }

// ==========================================
// FORWARD PASS
// ==========================================
int predictLocal(const float raw_features[FEATURE_COUNT], float* confidence) {
  float epsilon = 1e-7;
  for (size_t i = 0; i < FEATURE_COUNT; i++) {
    a0[i] = (raw_features[i] - norm_mean[i]) / sqrt(norm_var[i] + epsilon);
  }

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

  float maxLogit = -1e9;
  for (size_t j = 0; j < NUM_CLASSES; j++) {
    z3[j] = b3[j];
    for (size_t i = 0; i < L2_UNITS; i++) z3[j] += a2[i] * W3[i][j];
    if (z3[j] > maxLogit) maxLogit = z3[j];
  }

  float sumExp = 0.0f;
  for (size_t j = 0; j < NUM_CLASSES; j++) {
    a3[j] = exp(z3[j] - maxLogit);
    sumExp += a3[j];
  }

  int predClass = -1;
  float maxProb = -1.0;
  for (size_t j = 0; j < NUM_CLASSES; j++) {
    a3[j] /= sumExp;
    if (a3[j] > maxProb) {
      maxProb = a3[j];
      predClass = j;
    }
  }
  *confidence = maxProb;
  return predClass;
}

// ==========================================
// BACKWARD PASS (SGD)
// ==========================================
void trainLocalStep(const float raw_features[FEATURE_COUNT], int true_label) {
  float conf;
  predictLocal(raw_features, &conf);

  float dZ3[NUM_CLASSES], dZ2[L2_UNITS], dZ1[L1_UNITS];

  for (size_t j = 0; j < NUM_CLASSES; j++) {
    float y_onehot = (j == (size_t)true_label) ? 1.0f : 0.0f;
    dZ3[j] = a3[j] - y_onehot;
  }
  for (size_t i = 0; i < L2_UNITS; i++) {
    float error = 0.0f;
    for (size_t j = 0; j < NUM_CLASSES; j++) error += dZ3[j] * W3[i][j];
    dZ2[i] = error * relu_derivative(z2[i]);
  }
  for (size_t i = 0; i < L1_UNITS; i++) {
    float error = 0.0f;
    for (size_t j = 0; j < L2_UNITS; j++) error += dZ2[j] * W2[i][j];
    dZ1[i] = error * relu_derivative(z1[i]);
  }

  for (size_t j = 0; j < NUM_CLASSES; j++) {
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
// PROCESAR FEATURES RECIBIDAS
// ==========================================
void processReceivedFeatures(const char* payload, unsigned int length) {
  StaticJsonDocument<2048> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  
  if (err) {
    Serial.print("[BROKER] Error parseando JSON: ");
    Serial.println(err.c_str());
    return;
  }

  String senderID = doc["client_id"] | "unknown";
  JsonArray featArr = doc["features"].as<JsonArray>();
  
  if (featArr.size() != FEATURE_COUNT) {
    Serial.print("[BROKER] Error: se esperaban 36 features, llegaron ");
    Serial.println(featArr.size());
    return;
  }

  float features[FEATURE_COUNT];
  for (size_t i = 0; i < FEATURE_COUNT; i++) {
    features[i] = featArr[i].as<float>();
  }

  Serial.print("[BROKER] Features recibidas de ");
  Serial.println(senderID);

  // 1. INFERENCIA
  float confidence = 0;
  int pred = predictLocal(features, &confidence);
  
  Serial.print("[BROKER] Prediccion: ");
  Serial.print(CLASS_NAMES[pred]);
  Serial.print(" (clase ");
  Serial.print(pred);
  Serial.print(", confianza ");
  Serial.print(confidence * 100, 1);
  Serial.println("%)");

  // 2. ALERTA si no es benigno
  if (pred != 0) {
    totalAlertas++;
    Serial.println("========================================");
    Serial.print("  ATAQUE DETECTADO: ");
    Serial.println(CLASS_NAMES[pred]);
    Serial.println("========================================");

    StaticJsonDocument<512> alertDoc;
    alertDoc["alert"] = true;
    alertDoc["attack_type"] = CLASS_NAMES[pred];
    alertDoc["attack_class"] = pred;
    alertDoc["confidence"] = confidence;
    alertDoc["source"] = senderID;
    alertDoc["total_alerts"] = totalAlertas;

    char alertBuf[512];
    size_t alertLen = serializeJson(alertDoc, alertBuf);
    myBroker.publish(TOPIC_ALERTS, alertBuf, alertLen, false);
    Serial.println("[BROKER] Alerta publicada en fl/alerts");
  } else {
    Serial.println("[BROKER] Trafico BENIGNO - sin alerta.");
  }

  // 3. ENTRENAMIENTO LOCAL
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

  String json = "{";
  json += "\"client_id\":\"" + CLIENT_ID + "\",";
  json += "\"round\":" + String(currentRound) + ",";
  json += "\"num_samples\":" + String(numSamples) + ",";
  json += "\"weight_delta\":{\"W3\":[";

  for (size_t i = 0; i < L2_UNITS; i++) {
    json += "[";
    for (size_t j = 0; j < NUM_CLASSES; j++) {
      float delta = W3[i][j] - W3_base[i][j];
      json += String(delta, 5);
      if (j < NUM_CLASSES - 1) json += ",";
    }
    json += "]";
    if (i < L2_UNITS - 1) json += ",";
  }

  json += "],\"b3\":[";
  for (size_t j = 0; j < NUM_CLASSES; j++) {
    float delta = b3[j] - b3_base[j];
    json += String(delta, 5);
    if (j < NUM_CLASSES - 1) json += ",";
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

  Serial.println("========================================");
  Serial.println(" ESP32-S3 BROKER MQTT + IDS (6 clases)");
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
  delay(1);
}
