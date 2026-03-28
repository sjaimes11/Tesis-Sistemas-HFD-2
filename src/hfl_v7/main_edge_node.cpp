// =====================================================================
// main_edge_node.cpp — ESP32-S3 IoT Node (Inference Only) + HFL v7
// =====================================================================
// Rol:
// 1. Broker local (sMQTTBroker) para atrapar tráfico de sensores chicos.
// 2. Extracción de Features (13 variables).
// 3. Inferencia Local (Forward Pass con modelo Keras 13->32->16->8->3).
// 4. Envío de Features a Raspberry Pi Gateway (NO hace entrenamiento).
// 5. Recibe Actualización de Pesos del Gateway (fl/global_model).
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <sMQTTBroker.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "model_weights.h"
#include <math.h>

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48
#endif

// ==========================================
// CONFIG WiFi y MQTT GATEWAY
// ==========================================
const char* AP_SSID = "FL_SENSOR_NET";
const char* AP_PASS = "federated123";

// const char* STA_SSID = "TP-Link_AADB";       // <-- CAMBIAR POR TU RED
// const char* STA_PASS = "55707954";            // <-- CAMBIAR

const char* STA_SSID = "JAIMES_PUERTO 2.4";
const char* STA_PASS = "Anderson123";

const char* GATEWAY_MQTT_SERVER = "192.168.40.120"; // <-- IP Raspberry Pi 4 Gateway A
const int GATEWAY_MQTT_PORT = 1883;

// ==========================================
// TOPICS MQTT GATEWAY
// ==========================================
const char* TOPIC_FEATURES     = "fl/features";
const char* TOPIC_ALERTS       = "fl/alerts";
const char* TOPIC_GLOBAL_MODEL = "fl/global_model";
const String CLIENT_ID = "esp32_edge_node_A1";

// ==========================================
// MODELO MLP: 13 -> 32 -> 16 -> 8 -> 3
// ==========================================
constexpr size_t L1_UNITS = 32;
constexpr size_t L2_UNITS = 16;
constexpr size_t L3_UNITS = 8;
constexpr size_t OUTPUT_UNITS = NUM_CLASSES; // 3

const float ATTACK_THRESHOLD = 0.5f;
constexpr uint32_t MIN_PKTS_FOR_ML = 10;
constexpr uint32_t RULE_PKTS_ALERT = 200;

const char* CLASS_NAMES_STR[3] = {"normal", "mqtt_bruteforce", "scan_A"};

// Pesos mutables (Inferencia)
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

// ==========================================
// VENTANA DE MONITORIZACIÓN DE FLUJO
// ==========================================
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

WiFiClient wifiClient;
PubSubClient mqttGateway(wifiClient);

int totalAlertas = 0;
unsigned long ledOffTime = 0;

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
  out[7]=0; out[8]=0; out[9]=0;  // Banderas TCP/SYN/PSH no disponibles en simple MQTT
  out[10]=sqrtf(var_pkt);
  out[11]=(brokerMinPktLen<1e8f)?brokerMinPktLen:0;
  out[12]=brokerMaxPktLen;
}


// ==========================================
// BROKER MQTT EMBEBIDO (Local para Sensores)
// ==========================================
class MyBroker : public sMQTTBroker {
public:
  bool onEvent(sMQTTEvent *event) override {
    switch (event->Type()) {
      case NewClient_sMQTTEventType: {
        brokerConnections++;
        brokerTrackEvent(64);
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
        break;
      }
      default: break;
    }
    return true;
  }
};
MyBroker myBroker;

// ==========================================
// UTILIDADES
// ==========================================
float relu(float x) { return x > 0 ? x : 0.0f; }

void initModel() {
  for (size_t i=0; i<FEATURE_COUNT; i++) for (size_t j=0; j<L1_UNITS; j++) W1[i][j] = W1_base[i][j];
  for (size_t j=0; j<L1_UNITS; j++) b1[j] = b1_base[j];
  for (size_t i=0; i<L1_UNITS; i++) for (size_t j=0; j<L2_UNITS; j++) W2[i][j] = W2_base[i][j];
  for (size_t j=0; j<L2_UNITS; j++) b2[j] = b2_base[j];
  for (size_t i=0; i<L2_UNITS; i++) for (size_t j=0; j<L3_UNITS; j++) W3[i][j] = W3_base[i][j];
  for (size_t j=0; j<L3_UNITS; j++) b3[j] = b3_base[j];
  for (size_t i=0; i<L3_UNITS; i++) for (size_t j=0; j<OUTPUT_UNITS; j++) W4[i][j] = W4_base[i][j];
  for (size_t j=0; j<OUTPUT_UNITS; j++) b4[j] = b4_base[j];
  Serial.println("[MODEL] Base 13->32->16->8->3 cargado (Inferencia).");
}

void setLED(uint8_t r, uint8_t g, uint8_t b) { neopixelWrite(RGB_BUILTIN, r, g, b); }

// ==========================================
// INFERENCIA LOCAL (Forward Pass)
// ==========================================
int predictLocal(const float raw_features[FEATURE_COUNT], float* confidence) {
  // 1. Normalización
  for (size_t i=0; i<FEATURE_COUNT; i++)
    a0[i] = (raw_features[i] - scaler_mean[i]) / scaler_std[i];

  // 2. Capas Múltiples
  for (size_t j=0; j<L1_UNITS; j++) {
    z1[j] = b1[j];
    for(size_t i=0; i<FEATURE_COUNT; i++) z1[j] += a0[i]*W1[i][j];
    a1[j] = relu(z1[j]);
  }
  for (size_t j=0; j<L2_UNITS; j++) {
    z2[j] = b2[j];
    for (size_t i=0; i<L1_UNITS; i++) z2[j] += a1[i]*W2[i][j];
    a2[j] = relu(z2[j]);
  }
  for (size_t j=0; j<L3_UNITS; j++) {
    z3[j] = b3[j];
    for(size_t i=0; i<L2_UNITS; i++) z3[j] += a2[i]*W3[i][j];
    a3_val[j] = relu(z3[j]);
  }
  for(size_t j=0; j<OUTPUT_UNITS; j++){
    z4[j] = b4[j];
    for(size_t i=0; i<L3_UNITS; i++) z4[j] += a3_val[i]*W4[i][j];
  }

  // 3. Softmax
  float maxZ = z4[0];
  for(size_t j=1; j<OUTPUT_UNITS; j++) if(z4[j]>maxZ) maxZ = z4[j];
  float sumExp=0;
  for(size_t j=0; j<OUTPUT_UNITS; j++){
    a4[j] = expf(z4[j]-maxZ);
    sumExp += a4[j];
  }
  int predClass=0; float maxProb=0;
  for(size_t j=0; j<OUTPUT_UNITS; j++){
    a4[j] /= sumExp;
    if(a4[j]>maxProb){ maxProb=a4[j]; predClass=(int)j; }
  }
  *confidence = maxProb;
  return predClass;
}

// ==========================================
// ENVIAR FEATURES A GATEWAY (v7 - Sin entrenamiento ESP)
// ==========================================
void sendFeaturesToGateway(float features[FEATURE_COUNT]) {
  if (!mqttGateway.connected()) {
    if (mqttGateway.connect(CLIENT_ID.c_str())) {
      mqttGateway.subscribe(TOPIC_GLOBAL_MODEL);
    } else return;
  }
  
  StaticJsonDocument<512> doc;
  doc["client_id"] = CLIENT_ID;
  JsonArray array = doc.createNestedArray("features");
  for (size_t i = 0; i < FEATURE_COUNT; i++) array.add(features[i]);
  
  char buffer[512];
  size_t len = serializeJson(doc, buffer);
  
  if (mqttGateway.publish(TOPIC_FEATURES, buffer, len)) {
    Serial.println("[FL] Features enviadas crudas al Gateway.");
  }
}

// ==========================================
// IDS: ANALIZAR Y ALERTAR
// ==========================================
void analyzeAndAlert(float features[FEATURE_COUNT], bool ruleTriggered) {
  float confidence;
  int predClass = predictLocal(features, &confidence);

  Serial.print("[IDS] pkts="); Serial.print((int)features[0]);
  for (size_t j = 0; j < OUTPUT_UNITS; j++) {
    Serial.print(" | "); Serial.print(CLASS_NAMES_STR[j]); Serial.print("=");
    Serial.print(a4[j] * 100, 1); Serial.print("%");
  }
  Serial.print(" -> "); Serial.println(CLASS_NAMES_STR[predClass]);

  if (predClass != 0) {
    totalAlertas++;
    setLED(predClass == 1 ? 255 : 0, 0, predClass == 2 ? 255 : 0);
    ledOffTime = millis() + 3000;

    StaticJsonDocument<256> alertDoc;
    alertDoc["client_id"] = CLIENT_ID;
    alertDoc["alert"] = true;
    alertDoc["attack_type"] = CLASS_NAMES_STR[predClass];
    alertDoc["attack_probability"] = confidence;
    char alertBuf[256]; serializeJson(alertDoc, alertBuf);
    mqttGateway.publish(TOPIC_ALERTS, alertBuf);
  } else {
    setLED(0, 10, 0);
  }

  // SIEMPRE mandamos las features al Gateway para que él arme su Dataset y entrene
  if ((uint32_t)features[0] >= MIN_PKTS_FOR_ML || ruleTriggered) {
    sendFeaturesToGateway(features);
  }
}

// ==========================================
// CALLBACK MQTT: RECIBIR MODELO GLOBAL
// ==========================================
void onMqttGatewayCallback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) == TOPIC_GLOBAL_MODEL) {
    Serial.println("\n[FL] NUEVO MODELO GLOBAL RECIBIDO DEL GATEWAY!");
    DynamicJsonDocument doc(8192);
    DeserializationError err = deserializeJson(doc, payload, length);
    if (err) { Serial.println("[ERROR] Parseando JSON"); return; }
    
    JsonArray w3arr = doc["W3"].as<JsonArray>();
    JsonArray b3arr = doc["b3"].as<JsonArray>();
    JsonArray w4arr = doc["W4"].as<JsonArray>();
    JsonArray b4arr = doc["b4"].as<JsonArray>();

    // Sobreescribir directamente los pesos de inferencia local
    for (size_t i=0; i<L2_UNITS; i++) {
        JsonArray row = w3arr[i].as<JsonArray>();
        for(size_t j=0; j<L3_UNITS; j++) W3[i][j] = row[j].as<float>();
    }
    for (size_t j=0; j<L3_UNITS; j++) b3[j] = b3arr[j].as<float>();
    for (size_t i=0; i<L3_UNITS; i++) {
        JsonArray row = w4arr[i].as<JsonArray>();
        for(size_t j=0; j<OUTPUT_UNITS; j++) W4[i][j] = row[j].as<float>();
    }
    for (size_t j=0; j<OUTPUT_UNITS; j++) b4[j] = b4arr[j].as<float>();

    Serial.println("[FL] Pesos locales para inferencia actualizados.");
    setLED(30,30,0); ledOffTime = millis()+2000;
  }
}

// ==========================================
// SETUP & LOOP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(2000);
  setLED(0,10,0);
  
  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("[NODE] SENSOR AP IP: "); Serial.println(WiFi.softAPIP());

  WiFi.begin(STA_SSID, STA_PASS);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis()-t0<15000) { delay(500); Serial.print("."); }
  Serial.println("\n[NODE] Conectado a Wi-Fi (Gateway Network).");

  myBroker.init(1883);
  mqttGateway.setServer(GATEWAY_MQTT_SERVER, GATEWAY_MQTT_PORT);
  mqttGateway.setBufferSize(8192);
  mqttGateway.setCallback(onMqttGatewayCallback);

  initModel();
  resetBrokerFlow();
  brokerLastWindowMs = millis();
  Serial.println("[NODE] Iniciado HFL v7 - Inferencia Solamente.\n");
}

void loop() {
  myBroker.update(); // Flujo local simulado

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttGateway.connected()) {
      if (mqttGateway.connect(CLIENT_ID.c_str())) {
        mqttGateway.subscribe(TOPIC_GLOBAL_MODEL);
        Serial.println("[MQTT] Conectado a Gateway RPi 4.");
      }
    }
    mqttGateway.loop();
  }

  // Evaluar ventana
  if (millis() - brokerLastWindowMs >= BROKER_WINDOW_MS) {
    brokerLastWindowMs = millis();
    if (brokerGlobalPkts > 0) {
      float features[FEATURE_COUNT];
      brokerExtractFeatures(features);
      bool ruleTriggered = (brokerGlobalPkts >= RULE_PKTS_ALERT) || (brokerConnections >= RULE_PKTS_ALERT);
      
      if (brokerGlobalPkts >= MIN_PKTS_FOR_ML || ruleTriggered) {
         analyzeAndAlert(features, ruleTriggered);
      }
      resetBrokerFlow();
    }
  }
  
  if (ledOffTime>0 && millis()>ledOffTime) { setLED(0,10,0); ledOffTime=0; }
  delay(1);
}
