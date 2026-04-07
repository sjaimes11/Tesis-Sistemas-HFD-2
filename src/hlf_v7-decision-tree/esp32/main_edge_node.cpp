// =====================================================================
// main_edge_node.cpp — ESP32-S3 IoT Node + Decision Tree
// =====================================================================
// Flujo:
// 1. Broker local para atrapar tráfico MQTT de sensores.
// 2. Extracción de 13 features.
// 3. Inferencia local vía classify(...) usando model_weights.h.
// 4. Publicación opcional de features y alertas hacia un Gateway/RPi.
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

constexpr size_t FEATURE_COUNT = NUM_FEATURES;
const char* CLASS_NAMES_STR[NUM_CLASSES] = {"normal", "mqtt_bruteforce", "scan_A"};

const char* AP_SSID = "FL_SENSOR_NET";
const char* AP_PASS = "federated123";
const char* STA_SSID = "CAMBIAR_WIFI";
const char* STA_PASS = "CAMBIAR_PASSWORD";
const char* GATEWAY_MQTT_SERVER = "192.168.40.120";
const int GATEWAY_MQTT_PORT = 1883;

const char* TOPIC_FEATURES = "fl/features";
const char* TOPIC_ALERTS = "fl/alerts";
const String CLIENT_ID = "esp32_edge_node_classic";

constexpr uint32_t BROKER_WINDOW_MS = 5000;
constexpr uint32_t MIN_PKTS_FOR_ML = 10;

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

void setLED(uint8_t r, uint8_t g, uint8_t b) {
  neopixelWrite(RGB_BUILTIN, r, g, b);
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
    brokerSumIat += iat;
    brokerSumSqIat += iat * iat;
    if (iat < brokerMinIat) brokerMinIat = iat;
    if (iat > brokerMaxIat) brokerMaxIat = iat;
  } else {
    brokerFirstPktUs = now;
  }

  brokerGlobalPkts++;
  brokerGlobalBytes += pkt_len;
  brokerSumPktLen += (float)pkt_len;
  brokerSumSqPktLen += (float)pkt_len * pkt_len;
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
  out[3]  = (n > 1) ? brokerMinIat : 0;
  out[4]  = (n > 1) ? brokerMaxIat : 0;
  out[5]  = mean_pkt;
  out[6]  = (float)brokerGlobalBytes;
  out[7]  = 0;
  out[8]  = 0;
  out[9]  = 0;
  out[10] = sqrtf(var_pkt);
  out[11] = (brokerMinPktLen < 1e8f) ? brokerMinPktLen : 0;
  out[12] = brokerMaxPktLen;
}

class MyBroker : public sMQTTBroker {
public:
  bool onEvent(sMQTTEvent *event) override {
    switch (event->Type()) {
      case NewClient_sMQTTEventType:
        brokerConnections++;
        brokerTrackEvent(64);
        break;
      case LostConnect_sMQTTEventType:
        brokerTrackEvent(32);
        break;
      case Subscribe_sMQTTEventType:
        brokerTrackEvent(48);
        break;
      case Public_sMQTTEventType: {
        sMQTTPublicClientEvent *e = (sMQTTPublicClientEvent*)event;
        String topic = e->Topic().c_str();
        String payload = e->Payload().c_str();
        uint16_t msgSize = (uint16_t)(topic.length() + payload.length() + 8);
        brokerTrackEvent(msgSize);
        break;
      }
      default:
        break;
    }
    return true;
  }
};

MyBroker myBroker;

void sendFeaturesToGateway(const float features[FEATURE_COUNT], int predictedClass, float confidence) {
  if (!mqttGateway.connected()) {
    if (!mqttGateway.connect(CLIENT_ID.c_str())) return;
  }

  StaticJsonDocument<512> doc;
  doc["client_id"] = CLIENT_ID;
  doc["predicted_class"] = predictedClass;
  doc["predicted_label"] = CLASS_NAMES_STR[predictedClass];
  doc["confidence"] = confidence;
  JsonArray array = doc.createNestedArray("features");
  for (size_t i = 0; i < FEATURE_COUNT; i++) array.add(features[i]);

  char buffer[512];
  size_t len = serializeJson(doc, buffer);
  mqttGateway.publish(TOPIC_FEATURES, buffer, len);
}

void publishAlert(int predictedClass, float confidence) {
  if (!mqttGateway.connected()) return;
  StaticJsonDocument<256> doc;
  doc["client_id"] = CLIENT_ID;
  doc["alert"] = predictedClass != 0;
  doc["attack_type"] = CLASS_NAMES_STR[predictedClass];
  doc["attack_probability"] = confidence;

  char buffer[256];
  serializeJson(doc, buffer);
  mqttGateway.publish(TOPIC_ALERTS, buffer);
}

void analyzeAndPublish(float features[FEATURE_COUNT]) {
  float confidence = 0.0f;
  int predictedClass = classify(features, &confidence);

  Serial.print("[IDS] ");
  Serial.print(CLASS_NAMES_STR[predictedClass]);
  Serial.print(" | confidence=");
  Serial.println(confidence, 4);

  if (predictedClass == 0) setLED(0, 10, 0);
  else if (predictedClass == 1) setLED(255, 0, 0);
  else setLED(255, 0, 255);

  publishAlert(predictedClass, confidence);
  sendFeaturesToGateway(features, predictedClass, confidence);
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  setLED(0, 10, 0);

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(AP_SSID, AP_PASS);
  WiFi.begin(STA_SSID, STA_PASS);
  myBroker.init(1883);

  mqttGateway.setServer(GATEWAY_MQTT_SERVER, GATEWAY_MQTT_PORT);
  resetBrokerFlow();
  brokerLastWindowMs = millis();
  Serial.println("[NODE] Iniciado node clásico con inferencia local.");
}

void loop() {
  myBroker.update();

  if (WiFi.status() == WL_CONNECTED && !mqttGateway.connected()) {
    mqttGateway.connect(CLIENT_ID.c_str());
  }
  mqttGateway.loop();

  if (millis() - brokerLastWindowMs >= BROKER_WINDOW_MS) {
    brokerLastWindowMs = millis();
    if (brokerGlobalPkts >= MIN_PKTS_FOR_ML) {
      float features[FEATURE_COUNT];
      brokerExtractFeatures(features);
      analyzeAndPublish(features);
    }
    resetBrokerFlow();
  }
  delay(1);
}
