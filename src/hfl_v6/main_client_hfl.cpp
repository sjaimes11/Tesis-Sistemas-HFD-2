// =====================================================================
// main_client_hfl.cpp — ESP32-S3 Cliente IoT (Sensor + Traffic Gen)
// =====================================================================
// HFL v6: Genera ráfagas benignas (12-15 pkts) y simulaciones
// de ataque DoS (80-120 pkts) para probar el IDS 3-clase.
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* BROKER_AP_SSID = "FL_BROKER_NET";
const char* BROKER_AP_PASS = "federated123";
const char* MQTT_BROKER_IP = "192.168.4.1";
const int MQTT_PORT = 1883;

const char* TOPIC_SENSOR = "sensors/temperature";
const char* TOPIC_ALERTS = "fl/alerts";
const char* CLIENT_ID = "esp32_sensor_temp_01";

constexpr uint32_t PUBLISH_INTERVAL_MS = 6000;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

uint32_t lastPublishMs = 0;
int readingCount = 0;
int cycleCount = 0;

float baseTemp = 25.0f;

float readTemperature() {
    float drift = sin(millis() / 60000.0f) * 2.0f;
    float noise = ((float)random(-100, 100)) / 100.0f;
    return baseTemp + drift + noise;
}

float readHumidity() {
    float base = 60.0f;
    float drift = cos(millis() / 90000.0f) * 10.0f;
    float noise = ((float)random(-200, 200)) / 100.0f;
    return base + drift + noise;
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    if (String(topic) == TOPIC_ALERTS) {
        StaticJsonDocument<512> doc;
        DeserializationError err = deserializeJson(doc, payload, length);
        if (!err) {
            const char* attackType = doc["attack_type"] | "desconocido";
            float prob = doc["attack_probability"] | 0.0f;
            int predClass = doc["predicted_class"] | -1;

            Serial.println();
            Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            Serial.print("  ALERTA: "); Serial.println(attackType);
            Serial.print("  Confianza: "); Serial.print(prob * 100, 1); Serial.println("%");
            Serial.print("  Clase: "); Serial.println(predClass);
            Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            Serial.println();
        }
    }
}

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    WiFi.mode(WIFI_STA);
    WiFi.begin(BROKER_AP_SSID, BROKER_AP_PASS);
    Serial.print("[SENSOR] Conectando...");
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
        delay(500); Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("[SENSOR] IP: "); Serial.println(WiFi.localIP());
    }
}

void reconnectMQTT() {
    while (!mqttClient.connected()) {
        Serial.print("[SENSOR] MQTT...");
        if (mqttClient.connect(CLIENT_ID)) {
            Serial.println(" OK!");
            mqttClient.subscribe(TOPIC_ALERTS);
        } else {
            Serial.print(" fallo rc="); Serial.println(mqttClient.state());
            delay(5000);
        }
    }
}

void publishSingleReading(bool silent = false) {
    float temp = readTemperature();
    float hum  = readHumidity();
    readingCount++;

    StaticJsonDocument<256> doc;
    doc["client_id"] = CLIENT_ID;
    doc["reading"]   = readingCount;
    doc["temp_c"]    = round(temp * 10.0f) / 10.0f;
    doc["humidity"]  = round(hum * 10.0f) / 10.0f;
    doc["uptime_s"]  = millis() / 1000;

    char buffer[256];
    size_t len = serializeJson(doc, buffer);

    if (!silent) {
        Serial.print("[SENSOR] #"); Serial.print(readingCount);
        Serial.print(" | T:"); Serial.print(temp, 1); Serial.print("C");
        Serial.print(" ("); Serial.print(len); Serial.print("B)");
    }

    bool ok = mqttClient.publish(TOPIC_SENSOR, buffer);
    if (!silent) Serial.println(ok ? " OK" : " ERR");
}

void simulateTrafficCycle() {
    cycleCount++;
    bool isAttack = (cycleCount % 4 == 0);
    int packetsToSend = isAttack ? random(80, 120) : random(12, 16);
    int delayMs = isAttack ? 5 : 150;

    Serial.println("----------------------------------------");
    if (isAttack) {
        Serial.print("[SIM] ATAQUE: ");
    } else {
        Serial.print("[SIM] BENIGNO: ");
    }
    Serial.print(packetsToSend); Serial.println(" paquetes");

    for (int i = 0; i < packetsToSend; i++) {
        publishSingleReading(isAttack);
        delay(delayMs);
    }

    Serial.println("[SIM] Ciclo completado.");
    Serial.println("----------------------------------------");
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    randomSeed(analogRead(0));

    Serial.println("========================================");
    Serial.println(" ESP32-S3 SENSOR IoT (HFL v6)");
    Serial.println(" Rafagas benignas + DoS simulado");
    Serial.println("========================================");

    connectWiFi();
    mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
    mqttClient.setCallback(mqttCallback);
    mqttClient.setBufferSize(2048);
    lastPublishMs = millis();
    Serial.println("[SENSOR] Listo.\n");
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    if (!mqttClient.connected()) reconnectMQTT();
    mqttClient.loop();

    if (millis() - lastPublishMs >= PUBLISH_INTERVAL_MS) {
        lastPublishMs = millis();
        simulateTrafficCycle();
    }
    delay(10);
}
