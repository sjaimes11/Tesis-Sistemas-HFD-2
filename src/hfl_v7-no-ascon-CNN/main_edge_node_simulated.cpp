// =====================================================================
// main_edge_node_simulated.cpp — ESP32-S3 Edge Node (HFL v7-CNN)
// =====================================================================
// Rol:
// 1. Simulación Interna: Genera tráfico (Normal, Bruteforce, Scan_A).
// 2. Extracción de Features (13 variables).
// 3. Inferencia Local (TinyML) — Motor CNN-1D en C++:
//      Conv1D(32, k=3, same) -> Conv1D(16, k=3, same) -> GAP -> Dense8 -> Dense3
// 4. Envía Features a la Raspberry Pi (fl/features_plain) — JSON plano.
// 5. Recibe pesos Dense actualizados (fl/global_model_plain) — JSON plano.
//
// Probabilidades de simulación:
//   40% Normal | 30% MQTT Bruteforce | 30% Scan_A
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "model_weights.h"    // Generado por train_3class.py (CNN)
#include <math.h>

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48
#endif

// ==========================================
// CONFIG
// ==========================================
// const char* STA_SSID = "JAIMES_PUERTO 2.4"; // Reemplaza por el WiFi de tu casa
// const char* STA_PASS = "Anderson123";

const char* STA_SSID = "TP-Link_AADB";           // <-- CAMBIAR
const char* STA_PASS = "55707954";                // <-- CAMBIAR
const char* GATEWAY_MQTT_SERVER = "192.168.1.16"; // <-- IP Raspberry Pi 4
const int   GATEWAY_MQTT_PORT   = 1883;

const char* TOPIC_FEATURES     = "fl/features_plain";
const char* TOPIC_GLOBAL_MODEL = "fl/global_model_plain";
const String CLIENT_ID         = "esp32_edge_simulator_1";

// ==========================================
// ARQUITECTURA CNN-1D
// ==========================================
constexpr size_t CONV1_FILTERS = 32;
constexpr size_t CONV2_FILTERS = 16;
constexpr size_t DENSE1_UNITS  = 8;
constexpr size_t OUTPUT_UNITS  = 3;
constexpr size_t KERNEL_SIZE   = 3;
constexpr uint32_t RULE_PKTS_ALERT = 100;

const char* CLASS_NAMES_STR[3] = {"normal", "mqtt_bruteforce", "scan_A"};

// ── Pesos Dense (MUTABLES — actualizados por FL)
float Wd1[CONV2_FILTERS][DENSE1_UNITS]; float bd1[DENSE1_UNITS];
float Wdo[DENSE1_UNITS][OUTPUT_UNITS];  float bdo[OUTPUT_UNITS];

// ── Buffers de activaciones
float conv1_out[FEATURE_COUNT][CONV1_FILTERS];
float conv2_out[FEATURE_COUNT][CONV2_FILTERS];
float gap_out[CONV2_FILTERS];
float dense1_out[DENSE1_UNITS];
float softmax_out[OUTPUT_UNITS];

// ==========================================
// TRACKER DE FLUJO
// ==========================================
uint32_t      brokerGlobalPkts  = 0;
uint32_t      brokerGlobalBytes = 0;
float         brokerPshFlags    = 0;
unsigned long brokerLastPktUs   = 0;
float brokerSumIat=0,brokerSumSqIat=0,brokerMinIat=1e9f,brokerMaxIat=0;
float brokerSumPktLen=0,brokerSumSqPktLen=0,brokerMinPktLen=1e9f,brokerMaxPktLen=0;

WiFiClient   wifiClient;
PubSubClient mqttGateway(wifiClient);
int          totalAlertas     = 0;
unsigned long ledOffTime      = 0;
uint32_t      lastSimulationMs = 0;
uint32_t      msg_counter      = 0;

// ==========================================
// UTILIDADES
// ==========================================
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
void setLED(uint8_t r, uint8_t g, uint8_t b) { neopixelWrite(RGB_BUILTIN, r, g, b); }

void resetBrokerFlow() {
  brokerGlobalPkts=0; brokerGlobalBytes=0; brokerPshFlags=0; brokerLastPktUs=0;
  brokerSumIat=0; brokerSumSqIat=0; brokerMinIat=1e9f; brokerMaxIat=0;
  brokerSumPktLen=0; brokerSumSqPktLen=0; brokerMinPktLen=1e9f; brokerMaxPktLen=0;
}

void brokerTrackEvent(uint16_t pkt_len, bool is_psh) {
  unsigned long now = micros();
  if (brokerGlobalPkts > 0 && brokerLastPktUs > 0) {
    float iat = (now - brokerLastPktUs) / 1e6f;
    brokerSumIat+=iat; brokerSumSqIat+=iat*iat;
    if (iat < brokerMinIat) brokerMinIat=iat;
    if (iat > brokerMaxIat) brokerMaxIat=iat;
  }
  brokerGlobalPkts++; brokerGlobalBytes+=pkt_len;
  if (is_psh) brokerPshFlags+=1.0f;
  brokerSumPktLen+=(float)pkt_len; brokerSumSqPktLen+=(float)pkt_len*(float)pkt_len;
  if ((float)pkt_len < brokerMinPktLen) brokerMinPktLen=(float)pkt_len;
  if ((float)pkt_len > brokerMaxPktLen) brokerMaxPktLen=(float)pkt_len;
  brokerLastPktUs=now;
}

void brokerExtractFeatures(float out[FEATURE_COUNT]) {
  float n=(float)brokerGlobalPkts;
  for (size_t i=0;i<FEATURE_COUNT;i++) out[i]=0.0f;
  if (n<1.0f) return;
  float mean_pkt=brokerSumPktLen/n;
  float mean_iat=(n>1)?brokerSumIat/(n-1.0f):0.0f;
  float var_pkt=(n>1)?(brokerSumSqPktLen/n)-(mean_pkt*mean_pkt):0.0f;
  float var_iat=(n>1)?(brokerSumSqIat/(n-1.0f))-(mean_iat*mean_iat):0.0f;
  if (var_pkt<0) var_pkt=0; if (var_iat<0) var_iat=0;
  out[0]=n; out[1]=mean_iat; out[2]=sqrtf(var_iat);
  out[3]=(n>1)?brokerMinIat:0; out[4]=(n>1)?brokerMaxIat:0;
  out[5]=mean_pkt; out[6]=(float)brokerGlobalBytes;
  out[7]=brokerPshFlags; out[8]=0; out[9]=0;
  out[10]=sqrtf(var_pkt); out[11]=(brokerMinPktLen<1e8f)?brokerMinPktLen:0; out[12]=brokerMaxPktLen;
}

// ==========================================
// GENERADORES DE TRÁFICO SIMULADO
// Basado en estadísticas reales de los CSVs:
//   Normal:     pkts~5,   IAT~0.4ms,  pkt~63B,  PSH~2
//   Bruteforce: pkts~345, IAT~3.38s,  pkt~60B,  PSH~69
//   Scan_A:     pkts~1,   IAT~0,      pkt~44B,  PSH=0
// ==========================================
void generateNormalFeatures(float out[FEATURE_COUNT]) {
  int pkts = random(4, 9);
  Serial.print("\n--- [SIM] Normal ("); Serial.print(pkts); Serial.println(" pkts) ---");
  for (int i=0; i<pkts; i++) {
    uint16_t pkt_len;
    bool psh;
    if (i==0 || random(100)<40) { pkt_len=52; psh=false; }
    else                         { pkt_len=random(58,112); psh=(random(100)<60); }
    brokerTrackEvent(pkt_len, psh);
    delayMicroseconds(random(35, 680));
  }
  brokerExtractFeatures(out);
  resetBrokerFlow();
}

void generateBruteforceFeatures(float out[FEATURE_COUNT]) {
  float pkts    = (float)random(200, 500);
  float meanIat = random(100, 700) / 100.0f;
  float stdIat  = random(400, 1500) / 100.0f;
  float meanPkt = random(545, 650) / 10.0f;
  float psh     = pkts * (random(15, 25) / 100.0f);

  out[0]=pkts; out[1]=meanIat; out[2]=stdIat;
  out[3]=random(0,100)/100000.0f; out[4]=random(4000,12000)/100.0f;
  out[5]=meanPkt; out[6]=pkts*meanPkt;
  out[7]=psh; out[8]=0; out[9]=0;
  out[10]=random(20,70)/10.0f; out[11]=52.0f; out[12]=(float)random(60,90);

  Serial.print("\n>>> [SIM] MQTT Bruteforce (pkts="); Serial.print((int)pkts);
  Serial.print(", IAT="); Serial.print(meanIat,2); Serial.println("s) <<<");
}

void generateScanAFeatures(float out[FEATURE_COUNT]) {
  int pkts     = random(1, 4);
  float pktLen = (float)random(40, 48);

  out[0]=(float)pkts;
  out[1]=(pkts>1)?random(0,50)/100000.0f:0.0f;
  out[2]=(pkts>1)?random(0,30)/100000.0f:0.0f;
  out[3]=(pkts>1)?random(0,10)/100000.0f:0.0f;
  out[4]=(pkts>1)?random(0,80)/100000.0f:0.0f;
  out[5]=pktLen; out[6]=pktLen*pkts;
  out[7]=0; out[8]=(random(100)<40)?1.0f:0.0f; out[9]=0;
  out[10]=(pkts>1)?random(0,30)/10.0f:0.0f;
  out[11]=(float)random(40,46); out[12]=(float)random(40,52);

  Serial.print("\n$$$ [SIM] TCP Scan_A (pkts="); Serial.print(pkts);
  Serial.print(", pktLen="); Serial.print(pktLen,0); Serial.println(") $$$");
}

// ==========================================
// MOTOR CNN-1D
// ==========================================
void conv1d_same(const float input[][1], size_t T, size_t in_ch,
                 const float W[KERNEL_SIZE][1][CONV1_FILTERS],
                 const float bias[CONV1_FILTERS],
                 size_t out_filters,
                 float output[][CONV1_FILTERS]) {
  int pad = KERNEL_SIZE / 2;
  for (size_t t=0; t<T; t++) {
    for (size_t f=0; f<out_filters; f++) {
      float acc = bias[f];
      for (size_t k=0; k<KERNEL_SIZE; k++) {
        int src = (int)t + (int)k - pad;
        if (src>=0 && src<(int)T)
          for (size_t ic=0; ic<in_ch; ic++) acc += input[src][ic]*W[k][ic][f];
      }
      output[t][f] = relu(acc);
    }
  }
}

void conv1d_same_2(const float input[][CONV1_FILTERS], size_t T,
                   const float W[KERNEL_SIZE][CONV1_FILTERS][CONV2_FILTERS],
                   const float bias[CONV2_FILTERS],
                   float output[][CONV2_FILTERS]) {
  int pad = KERNEL_SIZE / 2;
  for (size_t t=0; t<T; t++) {
    for (size_t f=0; f<CONV2_FILTERS; f++) {
      float acc = bias[f];
      for (size_t k=0; k<KERNEL_SIZE; k++) {
        int src = (int)t + (int)k - pad;
        if (src>=0 && src<(int)T)
          for (size_t ic=0; ic<CONV1_FILTERS; ic++) acc += input[src][ic]*W[k][ic][f];
      }
      output[t][f] = relu(acc);
    }
  }
}

void initModel() {
  for (size_t i=0; i<CONV2_FILTERS; i++)
    for (size_t j=0; j<DENSE1_UNITS; j++) Wd1[i][j]=W_dense1[i][j];
  for (size_t j=0; j<DENSE1_UNITS; j++)  bd1[j]=b_dense1[j];
  for (size_t i=0; i<DENSE1_UNITS; i++)
    for (size_t j=0; j<OUTPUT_UNITS; j++) Wdo[i][j]=W_dense_out[i][j];
  for (size_t j=0; j<OUTPUT_UNITS; j++)  bdo[j]=b_dense_out[j];
  Serial.println("[CNN] Pesos cargados: Conv1+Conv2 (fijos) | Dense1+DenseOut (FL)");
}

int predictLocal(const float raw[FEATURE_COUNT], float* confidence) {
  float x_norm[FEATURE_COUNT][1];
  for (size_t i=0; i<FEATURE_COUNT; i++)
    x_norm[i][0] = (raw[i] - scaler_mean[i]) / scaler_std[i];

  conv1d_same(x_norm, FEATURE_COUNT, 1, W_conv1, b_conv1, CONV1_FILTERS, conv1_out);
  conv1d_same_2(conv1_out, FEATURE_COUNT, W_conv2, b_conv2, conv2_out);

  for (size_t f=0; f<CONV2_FILTERS; f++) {
    float s=0;
    for (size_t t=0; t<FEATURE_COUNT; t++) s+=conv2_out[t][f];
    gap_out[f]=s/(float)FEATURE_COUNT;
  }

  for (size_t j=0; j<DENSE1_UNITS; j++) {
    float acc=bd1[j];
    for (size_t i=0; i<CONV2_FILTERS; i++) acc+=gap_out[i]*Wd1[i][j];
    dense1_out[j]=relu(acc);
  }

  float z[OUTPUT_UNITS];
  for (size_t j=0; j<OUTPUT_UNITS; j++) {
    z[j]=bdo[j];
    for (size_t i=0; i<DENSE1_UNITS; i++) z[j]+=dense1_out[i]*Wdo[i][j];
  }
  float maxZ=z[0];
  for (size_t j=1; j<OUTPUT_UNITS; j++) if (z[j]>maxZ) maxZ=z[j];
  float sumE=0;
  for (size_t j=0; j<OUTPUT_UNITS; j++) { softmax_out[j]=expf(z[j]-maxZ); sumE+=softmax_out[j]; }
  int pred=0; float maxP=0;
  for (size_t j=0; j<OUTPUT_UNITS; j++) {
    softmax_out[j]/=sumE;
    if (softmax_out[j]>maxP) { maxP=softmax_out[j]; pred=(int)j; }
  }
  *confidence=maxP;
  return pred;
}

// ==========================================
// ENVÍO FEATURES (JSON plano)
// ==========================================
void sendFeaturesToGateway(float features[FEATURE_COUNT]) {
  if (!mqttGateway.connected()) return;

  StaticJsonDocument<512> doc;
  doc["client_id"] = CLIENT_ID;
  JsonArray arr = doc.createNestedArray("features");
  for (size_t i=0; i<FEATURE_COUNT; i++) arr.add(features[i]);

  char payload[512];
  size_t plen = serializeJson(doc, payload);
  msg_counter++;

  if (mqttGateway.publish(TOPIC_FEATURES, payload, plen)) {
    Serial.print("  [PLAIN] "); Serial.print(plen); Serial.println("B enviados");
  }
}

// ==========================================
// RECEPCIÓN MODELO GLOBAL (JSON plano)
// ==========================================
void onMqttGatewayCallback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) != TOPIC_GLOBAL_MODEL) return;

  Serial.println("\n[FL] ==============================================");
  Serial.println("[FL] NUEVO MODELO CNN RECIBIDO DEL GATEWAY (RPi)");

  DynamicJsonDocument doc(8192);
  if (deserializeJson(doc, payload, length)) { Serial.println("[ERROR] Parse JSON"); return; }
  Serial.print("[FL] [PLAIN] "); Serial.print(length); Serial.println("B recibidos OK");

  JsonArray wd1arr = doc["W_dense1"].as<JsonArray>();
  JsonArray bd1arr = doc["b_dense1"].as<JsonArray>();
  JsonArray wdoarr = doc["W_dense_out"].as<JsonArray>();
  JsonArray bdoarr = doc["b_dense_out"].as<JsonArray>();

  for (size_t i=0; i<CONV2_FILTERS; i++) {
    JsonArray row=wd1arr[i].as<JsonArray>();
    for (size_t j=0; j<DENSE1_UNITS; j++) Wd1[i][j]=row[j].as<float>();
  }
  for (size_t j=0; j<DENSE1_UNITS; j++)  bd1[j]=bd1arr[j].as<float>();
  for (size_t i=0; i<DENSE1_UNITS; i++) {
    JsonArray row=wdoarr[i].as<JsonArray>();
    for (size_t j=0; j<OUTPUT_UNITS; j++) Wdo[i][j]=row[j].as<float>();
  }
  for (size_t j=0; j<OUTPUT_UNITS; j++) bdo[j]=bdoarr[j].as<float>();

  Serial.println("[FL] Pesos Dense1+DenseOut CNN actualizados.");
  Serial.println("[FL] ==============================================");
  setLED(30,30,0); ledOffTime=millis()+2000;
}

// ==========================================
// SIMULACIÓN COMBINADA
// ==========================================
void simulateSelfTraffic() {
  int r = random(100);
  float features[FEATURE_COUNT];

  if (r < 40)      generateNormalFeatures(features);
  else if (r < 70) generateBruteforceFeatures(features);
  else             generateScanAFeatures(features);

  float conf;
  unsigned long t0=micros();
  int pred=predictLocal(features, &conf);
  unsigned long inf_us=micros()-t0;

  Serial.print("  [CNN] Pred ("); Serial.print(inf_us); Serial.print("us): ");
  for (size_t j=0; j<OUTPUT_UNITS; j++) {
    Serial.print(CLASS_NAMES_STR[j]); Serial.print("="); Serial.print(softmax_out[j]*100,1); Serial.print("% | ");
  }
  Serial.print(" => "); Serial.println(CLASS_NAMES_STR[pred]);

  if (pred != 0) {
    totalAlertas++;
    setLED(pred==1?255:0, 0, pred==2?255:0);
    ledOffTime=millis()+3000;
  } else {
    setLED(0, 10, 0);
  }
  sendFeaturesToGateway(features);
}

// ==========================================
// SETUP & LOOP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(2000);
  setLED(0,10,0);

  WiFi.mode(WIFI_STA);
  WiFi.begin(STA_SSID, STA_PASS);
  Serial.print("\n[NODE-CNN-SIM] Conectando a Wi-Fi...");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\n[NODE-CNN-SIM] Conectado.");

  mqttGateway.setServer(GATEWAY_MQTT_SERVER, GATEWAY_MQTT_PORT);
  mqttGateway.setBufferSize(8192);
  mqttGateway.setCallback(onMqttGatewayCallback);

  initModel();
  resetBrokerFlow();
  lastSimulationMs=millis();
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttGateway.connected()) {
      if (mqttGateway.connect(CLIENT_ID.c_str())) {
        mqttGateway.subscribe(TOPIC_GLOBAL_MODEL);
        Serial.println("[MQTT] Suscrito al Gateway (RPi).");
      }
    }
    mqttGateway.loop();
  }

  if (millis()-lastSimulationMs >= 5000) {
    lastSimulationMs=millis();
    simulateSelfTraffic();
  }

  if (ledOffTime>0 && millis()>ledOffTime) { setLED(0,10,0); ledOffTime=0; }
  delay(1);
}
