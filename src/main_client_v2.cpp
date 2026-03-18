// =====================================================================
// main_client.cpp - ESP32-S3 Cliente MQTT (Publicador de Features)
// =====================================================================
// Modelo: 13 features de flujo de red (MQTT-IoT-IDS2020)
// Clasificacion binaria: normal (0) / ataque (1)
// Se conecta al AP del Broker ESP32 y publica features cada 5 seg.
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ==========================================
// CONFIGURACION WiFi
// ==========================================
const char* BROKER_AP_SSID = "FL_BROKER_NET";
const char* BROKER_AP_PASS = "federated123";

const char* MQTT_BROKER_IP = "192.168.4.1";
const int MQTT_PORT = 1883;

const char* TOPIC_FEATURES = "fl/features";
const char* TOPIC_ALERTS   = "fl/alerts";

const char* CLIENT_ID = "esp32_client_01";

constexpr uint32_t WINDOW_MS = 5000;
constexpr size_t FEATURE_COUNT = 13;

static const char* FEATURE_NAMES[FEATURE_COUNT] = {
  "num_pkts", "mean_iat", "std_iat", "min_iat", "max_iat",
  "mean_pkt_len", "num_bytes", "num_psh_flags", "num_rst_flags",
  "num_urg_flags", "std_pkt_len", "min_pkt_len", "max_pkt_len"
};

WiFiClient espClient;
PubSubClient mqttClient(espClient);

uint32_t lastWindowMs = 0;
int windowCount = 0;

// ==========================================
// TRACKING DE FLUJO DE RED
// ==========================================
struct FlowStats {
  uint32_t num_pkts;
  uint32_t num_bytes;
  uint32_t num_psh;
  uint32_t num_rst;
  uint32_t num_urg;
  float    sum_pkt_len;
  float    sum_sq_pkt_len;
  float    min_pkt_len;
  float    max_pkt_len;
  unsigned long first_pkt_us;
  unsigned long last_pkt_us;
  float    sum_iat;
  float    sum_sq_iat;
  float    min_iat;
  float    max_iat;
};

FlowStats flow;

void resetFlow() {
  memset(&flow, 0, sizeof(flow));
  flow.min_pkt_len = 1e9f;
  flow.max_pkt_len = 0.0f;
  flow.min_iat     = 1e9f;
  flow.max_iat     = 0.0f;
  flow.first_pkt_us = 0;
  flow.last_pkt_us  = 0;
}

void trackPacket(uint16_t pkt_len, bool psh, bool rst, bool urg) {
  unsigned long now = micros();

  if (flow.num_pkts > 0 && flow.last_pkt_us > 0) {
    float iat = (now - flow.last_pkt_us) / 1e6f;
    flow.sum_iat    += iat;
    flow.sum_sq_iat += iat * iat;
    if (iat < flow.min_iat) flow.min_iat = iat;
    if (iat > flow.max_iat) flow.max_iat = iat;
  } else {
    flow.first_pkt_us = now;
  }

  flow.num_pkts++;
  flow.num_bytes += pkt_len;
  flow.sum_pkt_len    += (float)pkt_len;
  flow.sum_sq_pkt_len += (float)pkt_len * pkt_len;
  if ((float)pkt_len < flow.min_pkt_len) flow.min_pkt_len = (float)pkt_len;
  if ((float)pkt_len > flow.max_pkt_len) flow.max_pkt_len = (float)pkt_len;

  if (psh) flow.num_psh++;
  if (rst) flow.num_rst++;
  if (urg) flow.num_urg++;

  flow.last_pkt_us = now;
}

// ==========================================
// CALLBACK: Alertas del Broker
// ==========================================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) == TOPIC_ALERTS) {
    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, payload, length);
    if (!err) {
      const char* attackType = doc["attack_type"] | "desconocido";
      float prob = doc["attack_probability"] | 0.0f;
      const char* source = doc["source"] | "unknown";
      int totalAlerts = doc["total_alerts"] | 0;

      Serial.println();
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Serial.println("  ALERTA DE ATAQUE RECIBIDA");
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Serial.print("  Tipo: "); Serial.println(attackType);
      Serial.print("  Probabilidad de ataque: "); Serial.print(prob * 100, 1); Serial.println("%");
      Serial.print("  Fuente: "); Serial.println(source);
      Serial.print("  Alertas totales: "); Serial.println(totalAlerts);
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Serial.println();
    }
  }
}

// ==========================================
// CONEXION WiFi
// ==========================================
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(BROKER_AP_SSID, BROKER_AP_PASS);
  Serial.print("[CLIENT] Conectando a "); Serial.print(BROKER_AP_SSID);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(500); Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[CLIENT] Conectado! IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("[CLIENT] Timeout WiFi.");
  }
}

// ==========================================
// CONEXION MQTT
// ==========================================
void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("[CLIENT] Conectando MQTT...");
    if (mqttClient.connect(CLIENT_ID)) {
      Serial.println(" OK!");
      mqttClient.subscribe(TOPIC_ALERTS);
      Serial.println("[CLIENT] Suscrito a fl/alerts");
    } else {
      Serial.print(" fallo, rc="); Serial.print(mqttClient.state());
      Serial.println(" reintentando en 5s...");
      delay(5000);
    }
  }
}

// ==========================================
// EXTRAER FEATURES DE LA VENTANA
// ==========================================
void computeWindowFeatures(float out[FEATURE_COUNT]) {
  float n = (float)flow.num_pkts;

  if (n < 1.0f) {
    for (size_t i = 0; i < FEATURE_COUNT; i++) out[i] = 0.0f;
    windowCount++;
    return;
  }

  float mean_pkt_len = flow.sum_pkt_len / n;
  float mean_iat = (n > 1) ? flow.sum_iat / (n - 1.0f) : 0.0f;

  float var_pkt = (n > 1) ? (flow.sum_sq_pkt_len / n) - (mean_pkt_len * mean_pkt_len) : 0.0f;
  float var_iat = (n > 1) ? (flow.sum_sq_iat / (n - 1.0f)) - (mean_iat * mean_iat) : 0.0f;
  if (var_pkt < 0) var_pkt = 0;
  if (var_iat < 0) var_iat = 0;

  out[0]  = n;                                       // num_pkts
  out[1]  = mean_iat;                                // mean_iat
  out[2]  = sqrtf(var_iat);                          // std_iat
  out[3]  = (n > 1) ? flow.min_iat : 0.0f;          // min_iat
  out[4]  = (n > 1) ? flow.max_iat : 0.0f;          // max_iat
  out[5]  = mean_pkt_len;                            // mean_pkt_len
  out[6]  = (float)flow.num_bytes;                   // num_bytes
  out[7]  = (float)flow.num_psh;                     // num_psh_flags
  out[8]  = (float)flow.num_rst;                     // num_rst_flags
  out[9]  = (float)flow.num_urg;                     // num_urg_flags
  out[10] = sqrtf(var_pkt);                          // std_pkt_len
  out[11] = (flow.min_pkt_len < 1e8f) ? flow.min_pkt_len : 0.0f;  // min_pkt_len
  out[12] = flow.max_pkt_len;                        // max_pkt_len

  windowCount++;
}

// ==========================================
// PUBLICAR FEATURES
// ==========================================
void publishFeatures(const float features[FEATURE_COUNT]) {
  StaticJsonDocument<1024> doc;
  doc["client_id"] = CLIENT_ID;
  doc["window"] = windowCount;

  JsonArray arr = doc.createNestedArray("features");
  for (size_t i = 0; i < FEATURE_COUNT; i++) {
    arr.add(features[i]);
  }

  char buffer[1024];
  size_t len = serializeJson(doc, buffer);

  Serial.print("[CLIENT] Ventana ");
  Serial.print(windowCount);
  Serial.print(" | pkts="); Serial.print(flow.num_pkts);
  Serial.print(" bytes="); Serial.print(flow.num_bytes);
  Serial.print(" ("); Serial.print(len); Serial.print(" bytes JSON)...");

  bool ok = mqttClient.publish(TOPIC_FEATURES, buffer);
  Serial.println(ok ? " OK!" : " ERROR!");
}

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("========================================");
  Serial.println(" ESP32-S3 CLIENTE MQTT (13 features)");
  Serial.println(" Modelo: MQTT-IoT-IDS2020 binario");
  Serial.println("========================================");

  connectWiFi();
  mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(4096);

  resetFlow();
  lastWindowMs = millis();
  Serial.println("[CLIENT] Publicando features cada 5s...\n");
}

// ==========================================
// LOOP
// ==========================================
void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqttClient.connected()) reconnectMQTT();
  mqttClient.loop();

  // Simular tracking de cada paquete MQTT recibido/enviado
  // En produccion, hookear al callback de recepcion real
  trackPacket(64, true, false, false);

  if (millis() - lastWindowMs >= WINDOW_MS) {
    lastWindowMs = millis();
    float features[FEATURE_COUNT];
    computeWindowFeatures(features);
    publishFeatures(features);
    resetFlow();
  }
  delay(10);
}