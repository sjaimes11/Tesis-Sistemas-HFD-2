// =====================================================================
// main_client_scan_a.cpp — ESP32-S3 IoT (ATACANTE NMAP SCAN)
// =====================================================================
// HFL v6: Simula ser una herramienta de reconocimiento de red silenciosa.
// Genera inundaciones masivas de conexiones TCP vacías al puerto (SYN)
// y las aborta rápidamente, replicando las firmas de Nmap Aggressive
// sin enviar ni medio byte de payloads MQTT (sin PSH flags).
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>

const char* BROKER_AP_SSID = "FL_BROKER_NET";
const char* BROKER_AP_PASS = "federated123";
const char* MQTT_BROKER_IP = "192.168.4.1";
const int MQTT_PORT = 1883;

constexpr uint32_t PUBLISH_INTERVAL_MS = 5000;

WiFiClient rawClient;
uint32_t lastPublishMs = 0;

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    WiFi.mode(WIFI_STA);
    WiFi.begin(BROKER_AP_SSID, BROKER_AP_PASS);
    Serial.print("[ATACANTE] Asegurando enlace inalambrico...");
    while (WiFi.status() != WL_CONNECTED) { delay(100); Serial.print("."); }
    Serial.println(" LISTO.");
}

void simulateTCPScanA() {
    // Escaneo masivo de más de 120 intentos silenciosos
    int packets = random(120, 180);
    Serial.print("\n>>> [$$$] EXPLORANDO BROKER: NMAP TCP SCAN_A ("); 
    Serial.print(packets); Serial.println(" syn probes)");
    
    for (int i = 0; i < packets; i++) {
        // Intenta abrir el socket (TCP SYN seguido de Handshake) 
        // e inmediatamente lo cierra (FIN / RST) sin transaccionalidad
        if (rawClient.connect(MQTT_BROKER_IP, MQTT_PORT)) {
            rawClient.stop(); 
        }
        delay(3); // Inundación ultra-rápida (IAT ~3ms)
    }
    
    Serial.println("<<< [$$$] Exploracion terminada. Escáner oculto.");
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    randomSeed(analogRead(0));

    Serial.println("========================================");
    Serial.println("   ESP32 ATACANTE - 100% TCP SCAN A");
    Serial.println("========================================");

    connectWiFi();
    
    lastPublishMs = millis();
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    // No existe mqttClient.loop() porque ni siquiera instanciamos PubSubClient.
    // Esto es un ataque crudo de Capa 4 (Capa de Transporte).

    if (millis() - lastPublishMs >= PUBLISH_INTERVAL_MS) {
        lastPublishMs = millis();
        simulateTCPScanA();
    }
}
