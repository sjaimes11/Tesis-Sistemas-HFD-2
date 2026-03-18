// =====================================================================
// main_client_bruteforce.cpp — ESP32-S3 IoT (ATACANTE MQTT BRUTEFORCE)
// =====================================================================
// HFL v6: Simula ser un cliente hostil que ejecuta inundación continua
// de payloads MQTT repetitivos (simulando diccionarios de ataque o 
// estrés masivo a la capa de aplicación con banderas PSH activas).
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
const char* CLIENT_ID = "esp32_attacker_brute";

constexpr uint32_t PUBLISH_INTERVAL_MS = 5000;

WiFiClient espClient;
PubSubClient mqttClient(espClient);
uint32_t lastPublishMs = 0;

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    WiFi.mode(WIFI_STA);
    WiFi.begin(BROKER_AP_SSID, BROKER_AP_PASS);
    Serial.print("[ATACANTE] Abriendo conexion inalambrica...");
    while (WiFi.status() != WL_CONNECTED) { delay(100); Serial.print("."); }
    Serial.println(" RED COMPROMETIDA.");
}

void reconnectMQTT() {
    while (!mqttClient.connected() && WiFi.status() == WL_CONNECTED) {
        if (mqttClient.connect(CLIENT_ID)) {
            Serial.println("[ATACANTE] Socket MQTT incrustado.");
        } else {
            delay(500);
        }
    }
}

void simulateMQTTBruteforce() {
    int packets = random(100, 150);
    Serial.print("\n>>> [+++] INICIANDO ATAQUE MQTT BRUTEFORCE MASSIVO ("); 
    Serial.print(packets); Serial.println(" payloads)");
    
    if (!mqttClient.connected()) reconnectMQTT();
    
    // Payload JSON constante (diccionario falso)
    StaticJsonDocument<64> doc;
    doc["user"] = "admin";
    doc["pass"] = "123456";
    char buffer[64];
    serializeJson(doc, buffer);

    // Bucle de saturación a velocidad máxima del ESP32
    for (int i = 0; i < packets; i++) {
        mqttClient.publish(TOPIC_SENSOR, buffer);
        delay(2); // Retraso ultra-bajo (IAT ~ 2ms) para disparar la anomalía
    }
    
    Serial.println("<<< [-] Rafaga completada. Esperando recarga...");
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    randomSeed(analogRead(0));

    Serial.println("========================================");
    Serial.println("  ESP32 ATACANTE - 100% MQTT BRUTEFORCE");
    Serial.println("========================================");

    connectWiFi();
    mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
    
    lastPublishMs = millis();
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    mqttClient.loop();

    if (millis() - lastPublishMs >= PUBLISH_INTERVAL_MS) {
        lastPublishMs = millis();
        simulateMQTTBruteforce();
    }
}
