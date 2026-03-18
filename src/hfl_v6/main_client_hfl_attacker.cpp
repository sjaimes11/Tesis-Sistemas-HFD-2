// =====================================================================
// main_client_hfl_attacker.cpp — ESP32-S3 IoT (Atacante Agresivo)
// =====================================================================
// HFL v6: Simulación enfocada en ataques (más del 80% de las veces).
// Altera entre 3 comportamientos:
// 1. Normal (poca frecuencia): Tráfico de sensor benigno.
// 2. MQTT Bruteforce: Inundación de publicaciones MQTT (muchos PSH flags).
// 3. Scan A (TCP Flood): Inundación de conexiones de socket crudas
//    (sin payloads MQTT) para generar conexiones sin PSH flags.
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
const char* CLIENT_ID = "esp32_attacker_02";

constexpr uint32_t PUBLISH_INTERVAL_MS = 4000; // Más rápido para estresar el broker

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
            float prob = doc["attack_probability"] | 0.0f;
            Serial.println();
            Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            Serial.print("  ME HAN DETECTADO: "); Serial.println(attackType);
            Serial.print("  Certeza del IDS: "); Serial.print(prob * 100, 1); Serial.println("%");
            Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            Serial.println();
        }
    }
}

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    WiFi.mode(WIFI_STA);
    WiFi.begin(BROKER_AP_SSID, BROKER_AP_PASS);
    Serial.print("[ATACANTE] Conectando WiFi...");
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.print(" IP: "); Serial.println(WiFi.localIP());
}

void reconnectMQTT() {
    while (!mqttClient.connected() && WiFi.status() == WL_CONNECTED) {
        if (mqttClient.connect(CLIENT_ID)) {
            mqttClient.subscribe(TOPIC_ALERTS);
        } else {
            delay(2000);
        }
    }
}

void publishSingleReading(bool silent = false) {
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
    if (!silent) {
        Serial.print("[NORMAL] Pkt #"); Serial.println(readingCount);
    }
}

// Genera un ataque tipo Bruteforce/Flood llenando la red con mensajes MQTT pesados
void simulateMQTTBruteforce() {
    int packets = random(100, 150);
    Serial.print(">>> [ATAQUE] MQTT Bruteforce Flood ("); Serial.print(packets); Serial.println(" pkts)...");
    
    if (!mqttClient.connected()) reconnectMQTT();
    
    for (int i = 0; i < packets; i++) {
        publishSingleReading(true);
        delay(2); // Muy bajo IAT
    }
    Serial.println("<<< [ATAQUE] Fin de Bruteforce.");
}

// Genera un ataque tipo Scan_A (Inundación de conexiones de sockets sin payload PSH)
void simulateTCPScanA() {
    int packets = random(100, 150);
    Serial.print(">>> [ATAQUE] TCP Scan_A / SYN Flood ("); Serial.print(packets); Serial.println(" conexiones)...");
    
    // Nos desconectamos primero del broker MQTT para no interferir con las alertas,
    // y hacemos conexiones crudas usando WiFiClient directo.
    if(mqttClient.connected()) mqttClient.disconnect();

    WiFiClient rawClient;
    for (int i = 0; i < packets; i++) {
        // Intenta conectar y cerrar muy rápido
        if (rawClient.connect(MQTT_BROKER_IP, MQTT_PORT)) {
            rawClient.stop(); // Cierra agresivamente sin enviar payload MQTT
        }
        delay(3); // Muy bajo IAT
    }
    Serial.println("<<< [ATAQUE] Fin de Scan_A TCP.");
}

void simulateBenignTraffic() {
    int packets = random(10, 20);
    Serial.print("--- [SIM] Trafico Normal ("); Serial.print(packets); Serial.println(" pkts)...");
    if (!mqttClient.connected()) reconnectMQTT();
    
    for (int i = 0; i < packets; i++) {
        publishSingleReading(false);
        delay(150); // IAT alto y normal
    }
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    randomSeed(analogRead(0));

    Serial.println("========================================");
    Serial.println(" SENSOR ATACANTE HFL v6 (MALICIOSO)");
    Serial.println(" 20% Normal | 40% Bruteforce | 40% Scan");
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
        
        int r = random(100);
        if (r < 20) {
            // 20% de probabilidad: comportarse bien
            simulateBenignTraffic();
        } else if (r < 60) {
            // 40% de probabilidad: Ataque MQTT
            simulateMQTTBruteforce();
        } else {
            // 40% de probabilidad: Ataque de escaner de red
            simulateTCPScanA();
        }
    }
    delay(10);
}
