// =====================================================================
// main_client.cpp - ESP32-S3 Cliente MQTT (Sensor de Temperatura)
// =====================================================================
// Simula un sensor IoT que publica temperatura.
// Se ha modificado para enviar ráfagas de paquetes simulando tráfico
// "Normal" (10-15 paquetes) y periódicamente lanzar un "Ataque DoS"
// local (50+ paquetes rápidos) para activar el IDS en el Broker.
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

// Intervalo base para el ciclo de tráfico (cada 6 segundos)
constexpr uint32_t PUBLISH_INTERVAL_MS = 6000;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

uint32_t lastPublishMs = 0;
int readingCount = 0;
int cycleCount = 0; // Para saber cuándo lanzar ataque

// ==========================================
// SIMULACION DEL SENSOR DE TEMPERATURA
// ==========================================
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
// PUBLICAR LECTURA INDIVIDUAL
// ==========================================
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
        Serial.print(" | Temp: "); Serial.print(temp, 1); Serial.print("C");
        Serial.print(" ("); Serial.print(len); Serial.print("B)...");
    }

    bool ok = mqttClient.publish(TOPIC_SENSOR, buffer);
    if (!silent) {
        Serial.println(ok ? " OK" : " ERROR");
    }
}

// ==========================================
// SIMULAR TRAFICO (NORMAL O ATAQUE)
// ==========================================
void simulateTrafficCycle() {
    cycleCount++;
    
    // Cada 4 ciclos (aprox 24s), simulamos un "Ataque DoS"
    bool isAttack = (cycleCount % 4 == 0);
    
    // Si no es ataque, enviamos ~12-15 paquetes (Normal)
    // Si es ataque, enviamos ~100 paquetes rapidos (DoS Flood)
    int packetsToSend = isAttack ? random(80, 120) : random(12, 16);
    int delayBetwenPkts = isAttack ? 5 : 150; 
    
    Serial.println("----------------------------------------");
    if (isAttack) {
        Serial.print("[SIMULACION] Lanzando ráfaga de ATAQUE: ");
    } else {
        Serial.print("[SIMULACION] Enviando tráfico BENIGNO: ");
    }
    Serial.print(packetsToSend);
    Serial.println(" paquetes...");
    
    for (int i = 0; i < packetsToSend; i++) {
        // En un ataque, no queremos imprimir TODO en la consola para no bloquear el ESP
        bool silent = isAttack; 
        publishSingleReading(silent);
        delay(delayBetwenPkts); 
    }
    
    Serial.println("[SIMULACION] Ciclo completado.");
    Serial.println("----------------------------------------");
}

// ==========================================
// SETUP
// ==========================================
void setup() {
    Serial.begin(115200);
    delay(2000);
    randomSeed(analogRead(0));

    Serial.println("========================================");
    Serial.println(" ESP32-S3 SENSOR IoT - TRAFFIC GENERATOR");
    Serial.println(" Genera ráfagas benignas (12-15 pkts) y");
    Serial.println(" simulaciones de DoS (80-120 pkts).");
    Serial.println("========================================");

    connectWiFi();
    mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
    mqttClient.setCallback(mqttCallback);
    // Aumentamos buffer para aguantar ráfagas
    mqttClient.setBufferSize(2048); 

    lastPublishMs = millis();
    Serial.println("[SENSOR] Listo. Iniciando ciclos...\n");
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
        simulateTrafficCycle();
    }

    delay(10);
}