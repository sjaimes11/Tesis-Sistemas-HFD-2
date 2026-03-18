// =====================================================================
// main_client_normal.cpp — ESP32-S3 IoT (CLIENTE BENIGNO)
// =====================================================================
// HFL v6: Simula el comportamiento ideal de un sensor IoT.
// Sólo envía ráfagas espaciadas de datos telemétricos normales.
// Escucha alertas por MQTT y las imprime (reacción a intrusos).
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
const char* CLIENT_ID = "esp32_sensor_normal";

constexpr uint32_t PUBLISH_INTERVAL_MS = 6000;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

uint32_t lastPublishMs = 0;
int readingCount = 0;

float readTemperature() { return 25.0f + sin(millis() / 60000.0f) * 2.0f; }
float readHumidity() { return 60.0f + cos(millis() / 90000.0f) * 10.0f; }

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    if (String(topic) == TOPIC_ALERTS) {
        StaticJsonDocument<512> doc;
        DeserializationError err = deserializeJson(doc, payload, length);
        if (!err) {
            const char* attackType = doc["attack_type"] | "desconocido";
            Serial.println("\n[SISTEMA] !!! ALERTA DE SEGURIDAD RECIBIDA !!!");
            Serial.print("  El Broker aislo un atacante tipo: "); Serial.println(attackType);
        }
    }
}

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    WiFi.mode(WIFI_STA);
    WiFi.begin(BROKER_AP_SSID, BROKER_AP_PASS);
    Serial.print("[SENSOR] Conectando WiFi...");
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.print(" IP: "); Serial.println(WiFi.localIP());
}

void reconnectMQTT() {
    while (!mqttClient.connected() && WiFi.status() == WL_CONNECTED) {
        if (mqttClient.connect(CLIENT_ID)) {
            mqttClient.subscribe(TOPIC_ALERTS);
            Serial.println("[SENSOR] Broker MQTT Conectado. Modo 100% Benigno.");
        } else {
            delay(2000);
        }
    }
}

void publishBenignTraffic() {
    int packets = random(12, 16);
    Serial.print("\n--- [NORM] Enviando Lote de Telemetria ("); Serial.print(packets); Serial.println(" pkts) ---");
    
    if (!mqttClient.connected()) reconnectMQTT();
    
    for (int i = 0; i < packets; i++) {
        float temp = readTemperature();
        float hum  = readHumidity();
        readingCount++;

        StaticJsonDocument<128> doc;
        doc["id"] = CLIENT_ID;
        doc["t"]  = round(temp * 10.0f) / 10.0f;
        doc["h"]  = round(hum * 10.0f) / 10.0f;

        char buffer[128];
        serializeJson(doc, buffer);
        mqttClient.publish(TOPIC_SENSOR, buffer);
        
        Serial.print("    [+] Pkt #"); Serial.print(readingCount); 
        Serial.print(" | T: "); Serial.print(temp, 1); Serial.println("C Enviado.");
        delay(150); // IAT seguro y benigno (~150ms)
    }
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    randomSeed(analogRead(0));

    Serial.println("========================================");
    Serial.println("  ESP32 SENSOR - 100% TRAFICO NORMAL");
    Serial.println("========================================");

    connectWiFi();
    mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
    mqttClient.setCallback(mqttCallback);
    
    lastPublishMs = millis();
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    mqttClient.loop();

    if (millis() - lastPublishMs >= PUBLISH_INTERVAL_MS) {
        lastPublishMs = millis();
        publishBenignTraffic();
    }
    delay(10);
}
