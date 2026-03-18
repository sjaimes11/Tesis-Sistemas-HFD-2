// =====================================================================
// main_client.cpp - ESP32-S3 Cliente MQTT (Sensor de Temperatura)
// =====================================================================
// Simula un sensor IoT que publica temperatura cada 5 segundos.
// El broker monitorea este trafico como "normal" y detecta anomalias
// cuando un atacante (JMeter) inunda la red.
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

const char* TOPIC_SENSOR    = "sensors/temperature";
const char* TOPIC_ALERTS    = "fl/alerts";

const char* CLIENT_ID = "esp32_sensor_temp_01";

constexpr uint32_t PUBLISH_INTERVAL_MS = 5000;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

uint32_t lastPublishMs = 0;
int readingCount = 0;

// ==========================================
// SIMULACION DEL SENSOR DE TEMPERATURA
// ==========================================
float baseTemp = 25.0f;

float readTemperature() {
    // Simula un sensor DHT22 / DS18B20
    // Temperatura base + ruido gaussiano + deriva lenta
    float drift = sin(millis() / 60000.0f) * 2.0f;  // +-2C cada minuto
    float noise = ((float)random(-100, 100)) / 100.0f;  // +-1C ruido
    return baseTemp + drift + noise;
}

float readHumidity() {
    float base = 60.0f;
    float drift = cos(millis() / 90000.0f) * 10.0f;
    float noise = ((float)random(-200, 200)) / 100.0f;
    return base + drift + noise;
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
            Serial.println("  ALERTA DE ATAQUE RECIBIDA DEL BROKER");
            Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            Serial.print("  Tipo: "); Serial.println(attackType);
            Serial.print("  Probabilidad: "); Serial.print(prob * 100, 1); Serial.println("%");
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
    Serial.print("[SENSOR] Conectando a "); Serial.print(BROKER_AP_SSID);
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
        delay(500); Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("[SENSOR] Conectado! IP: "); Serial.println(WiFi.localIP());
    } else {
        Serial.println("[SENSOR] Timeout WiFi.");
    }
}

// ==========================================
// CONEXION MQTT
// ==========================================
void reconnectMQTT() {
    while (!mqttClient.connected()) {
        Serial.print("[SENSOR] Conectando MQTT...");
        if (mqttClient.connect(CLIENT_ID)) {
            Serial.println(" OK!");
            mqttClient.subscribe(TOPIC_ALERTS);
            Serial.println("[SENSOR] Suscrito a fl/alerts");
        } else {
            Serial.print(" fallo, rc="); Serial.print(mqttClient.state());
            Serial.println(" reintentando en 5s...");
            delay(5000);
        }
    }
}

// ==========================================
// PUBLICAR LECTURA DEL SENSOR
// ==========================================
void publishSensorData() {
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

    Serial.print("[SENSOR] #"); Serial.print(readingCount);
    Serial.print(" | Temp: "); Serial.print(temp, 1); Serial.print("C");
    Serial.print(" | Hum: "); Serial.print(hum, 1); Serial.print("%");
    Serial.print(" ("); Serial.print(len); Serial.print("B)...");

    bool ok = mqttClient.publish(TOPIC_SENSOR, buffer);
    Serial.println(ok ? " OK" : " ERROR");
}

// ==========================================
// SETUP
// ==========================================
void setup() {
    Serial.begin(115200);
    delay(2000);
    randomSeed(analogRead(0));

    Serial.println("========================================");
    Serial.println(" ESP32-S3 SENSOR IoT (Temperatura)");
    Serial.println(" Publica cada 5s en sensors/temperature");
    Serial.println(" Escucha alertas en fl/alerts");
    Serial.println("========================================");

    connectWiFi();
    mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
    mqttClient.setCallback(mqttCallback);
    mqttClient.setBufferSize(1024);

    lastPublishMs = millis();
    Serial.println("[SENSOR] Listo. Publicando cada 5s...\n");
}

// ==========================================
// LOOP
// ==========================================
void loop() {
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    if (!mqttClient.connected()) reconnectMQTT();
    mqttClient.loop();

    if (millis() - lastPublishMs >= PUBLISH_INTERVAL_MS) {
        lastPublishMs = millis();
        publishSensorData();
    }

    delay(10);
}