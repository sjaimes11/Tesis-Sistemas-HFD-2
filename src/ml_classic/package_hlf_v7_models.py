"""
=============================================================================
 package_hlf_v7_models.py — Empaqueta bundles ESP32/Raspberry por modelo
=============================================================================
 Genera una carpeta `hlf_v7-<modelo>` por cada modelo clásico entrenado:

 - `esp32/model_weights.h` con el export C listo para usar
 - `esp32/main_edge_node.cpp` con inferencia local usando `classify(...)`
 - `raspberry/model.pkl` + `scaler.pkl` (si existe)
 - `raspberry/mqtt_gateway.py` para inferencia vía MQTT
 - `raspberry/predict_local.py` para pruebas/offline
 - metadatos (`feature_order.csv`, `label_map.json`, `scaler_params.json`)
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from textwrap import dedent

try:
    from .export_to_c import export_model
except ImportError:
    from export_to_c import export_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTEFACT_ROOT = REPO_ROOT / "ml_outputs"

MODEL_SPECS = [
    {
        "artifact_dir": "decision_tree_model",
        "bundle_name": "hlf_v7-decision-tree",
        "display_name": "Decision Tree",
        "model_file": "decision_tree_best.pkl",
        "scaler_file": None,
        "supports_proba": True,
    },
    {
        "artifact_dir": "logistic_regression_model",
        "bundle_name": "hlf_v7-logistic-regression",
        "display_name": "Logistic Regression",
        "model_file": "logistic_regression_best.pkl",
        "scaler_file": "scaler_lr.pkl",
        "supports_proba": True,
    },
    {
        "artifact_dir": "random_forest_model",
        "bundle_name": "hlf_v7-random-forest",
        "display_name": "Random Forest",
        "model_file": "random_forest_best.pkl",
        "scaler_file": None,
        "supports_proba": True,
    },
    {
        "artifact_dir": "svm_model",
        "bundle_name": "hlf_v7-svm",
        "display_name": "SVM",
        "model_file": "svm_selected.pkl",
        "scaler_file": "scaler_svm.pkl",
        "supports_proba": False,
    },
]


def read_feature_order(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    values: list[str] = []
    for row in rows:
        if not row:
            continue
        value = row[0].strip()
        if not value or value == "feature":
            continue
        values.append(value)
    return values


def read_label_map(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [data[str(index)] for index in sorted((int(key) for key in data.keys()))]


def safe_reset_dir(path: Path, allowed_root: Path) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if allowed_root not in path.parents:
        raise ValueError(f"Ruta fuera del workspace esperado: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def render_bundle_readme(display_name: str, class_names: list[str], feature_names: list[str], header_kb: float, warnings: list[str]) -> str:
    warnings_md = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- Sin advertencias adicionales."
    return dedent(
        f"""\
        # {display_name} — Bundle `hlf_v7`

        Este paquete contiene una versión de despliegue para:
        - ESP32: `esp32/model_weights.h` y `esp32/main_edge_node.cpp`
        - Raspberry Pi: `raspberry/model.pkl`, `raspberry/mqtt_gateway.py`, `raspberry/predict_local.py`

        ## Clases
        {chr(10).join(f"- `{index}` = `{name}`" for index, name in enumerate(class_names))}

        ## Features
        {chr(10).join(f"- `{name}`" for name in feature_names)}

        ## Tamaño del header ESP32
        - `model_weights.h`: {header_kb:.1f} KB

        ## Advertencias
        {warnings_md}

        ## Uso rápido
        - ESP32: copia `esp32/model_weights.h` y `esp32/main_edge_node.cpp` a tu sketch/firmware.
        - Raspberry Pi: instala `raspberry/requirements.txt` y ejecuta `python raspberry/mqtt_gateway.py`.
        """
    )


def render_esp32_main(class_names: list[str], display_name: str) -> str:
    class_array = ", ".join(f'"{name}"' for name in class_names)
    return dedent(
        f"""\
        // =====================================================================
        // main_edge_node.cpp — ESP32-S3 IoT Node + {display_name}
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
        const char* CLASS_NAMES_STR[NUM_CLASSES] = {{{class_array}}};

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

        void setLED(uint8_t r, uint8_t g, uint8_t b) {{
          neopixelWrite(RGB_BUILTIN, r, g, b);
        }}

        void resetBrokerFlow() {{
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
        }}

        void brokerTrackEvent(uint16_t pkt_len) {{
          unsigned long now = micros();
          if (brokerGlobalPkts > 0 && brokerLastPktUs > 0) {{
            float iat = (now - brokerLastPktUs) / 1e6f;
            brokerSumIat += iat;
            brokerSumSqIat += iat * iat;
            if (iat < brokerMinIat) brokerMinIat = iat;
            if (iat > brokerMaxIat) brokerMaxIat = iat;
          }} else {{
            brokerFirstPktUs = now;
          }}

          brokerGlobalPkts++;
          brokerGlobalBytes += pkt_len;
          brokerSumPktLen += (float)pkt_len;
          brokerSumSqPktLen += (float)pkt_len * pkt_len;
          if ((float)pkt_len < brokerMinPktLen) brokerMinPktLen = (float)pkt_len;
          if ((float)pkt_len > brokerMaxPktLen) brokerMaxPktLen = (float)pkt_len;
          brokerLastPktUs = now;
        }}

        void brokerExtractFeatures(float out[FEATURE_COUNT]) {{
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
        }}

        class MyBroker : public sMQTTBroker {{
        public:
          bool onEvent(sMQTTEvent *event) override {{
            switch (event->Type()) {{
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
              case Public_sMQTTEventType: {{
                sMQTTPublicClientEvent *e = (sMQTTPublicClientEvent*)event;
                String topic = e->Topic().c_str();
                String payload = e->Payload().c_str();
                uint16_t msgSize = (uint16_t)(topic.length() + payload.length() + 8);
                brokerTrackEvent(msgSize);
                break;
              }}
              default:
                break;
            }}
            return true;
          }}
        }};

        MyBroker myBroker;

        void sendFeaturesToGateway(const float features[FEATURE_COUNT], int predictedClass, float confidence) {{
          if (!mqttGateway.connected()) {{
            if (!mqttGateway.connect(CLIENT_ID.c_str())) return;
          }}

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
        }}

        void publishAlert(int predictedClass, float confidence) {{
          if (!mqttGateway.connected()) return;
          StaticJsonDocument<256> doc;
          doc["client_id"] = CLIENT_ID;
          doc["alert"] = predictedClass != 0;
          doc["attack_type"] = CLASS_NAMES_STR[predictedClass];
          doc["attack_probability"] = confidence;

          char buffer[256];
          serializeJson(doc, buffer);
          mqttGateway.publish(TOPIC_ALERTS, buffer);
        }}

        void analyzeAndPublish(float features[FEATURE_COUNT]) {{
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
        }}

        void setup() {{
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
        }}

        void loop() {{
          myBroker.update();

          if (WiFi.status() == WL_CONNECTED && !mqttGateway.connected()) {{
            mqttGateway.connect(CLIENT_ID.c_str());
          }}
          mqttGateway.loop();

          if (millis() - brokerLastWindowMs >= BROKER_WINDOW_MS) {{
            brokerLastWindowMs = millis();
            if (brokerGlobalPkts >= MIN_PKTS_FOR_ML) {{
              float features[FEATURE_COUNT];
              brokerExtractFeatures(features);
              analyzeAndPublish(features);
            }}
            resetBrokerFlow();
          }}
          delay(1);
        }}
        """
    )


def render_raspberry_predict(display_name: str) -> str:
    return dedent(
        f"""\
        import argparse
        import csv
        import json
        from pathlib import Path

        import joblib
        import numpy as np


        BASE_DIR = Path(__file__).resolve().parent
        MODEL_PATH = BASE_DIR / "model.pkl"
        SCALER_PATH = BASE_DIR / "scaler.pkl"
        FEATURE_ORDER_PATH = BASE_DIR / "feature_order.csv"
        LABEL_MAP_PATH = BASE_DIR / "label_map.json"


        def load_feature_order():
            with FEATURE_ORDER_PATH.open("r", encoding="utf-8", newline="") as handle:
                values = [row[0].strip() for row in csv.reader(handle) if row]
            return [value for value in values if value and value != "feature"]


        def load_label_names():
            data = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
            return [data[str(index)] for index in sorted((int(key) for key in data.keys()))]


        def softmax(x):
            x = np.asarray(x, dtype=np.float64)
            x = x - np.max(x)
            exps = np.exp(x)
            return exps / np.sum(exps)


        def main():
            parser = argparse.ArgumentParser(description="{display_name} predictor para Raspberry Pi")
            parser.add_argument("--features-json", required=True, help="Lista JSON con las 13 features")
            args = parser.parse_args()

            features = np.asarray(json.loads(args.features_json), dtype=np.float32).reshape(1, -1)
            feature_names = load_feature_order()
            label_names = load_label_names()

            if features.shape[1] != len(feature_names):
                raise ValueError(f"Se esperaban {{len(feature_names)}} features y llegaron {{features.shape[1]}}")

            model = joblib.load(MODEL_PATH)
            if SCALER_PATH.exists():
                scaler = joblib.load(SCALER_PATH)
                features = scaler.transform(features)

            pred = int(model.predict(features)[0])
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(features)[0]))
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(features)
                confidence = float(np.max(softmax(decision[0] if decision.ndim > 1 else [0.0, decision[0]])))
            else:
                confidence = None

            result = {{
                "predicted_class": pred,
                "predicted_label": label_names[pred],
                "confidence": confidence,
            }}
            print(json.dumps(result, indent=2))


        if __name__ == "__main__":
            main()
        """
    )


def render_raspberry_gateway(display_name: str) -> str:
    return dedent(
        f"""\
        import csv
        import json
        from pathlib import Path

        import joblib
        import numpy as np
        import paho.mqtt.client as mqtt


        BASE_DIR = Path(__file__).resolve().parent
        MODEL_PATH = BASE_DIR / "model.pkl"
        SCALER_PATH = BASE_DIR / "scaler.pkl"
        FEATURE_ORDER_PATH = BASE_DIR / "feature_order.csv"
        LABEL_MAP_PATH = BASE_DIR / "label_map.json"

        MQTT_BROKER = "localhost"
        MQTT_PORT = 1883
        TOPIC_FEATURES = "fl/features"
        TOPIC_ALERTS = "fl/alerts"
        TOPIC_PREDICTIONS = "fl/predictions"


        def load_feature_order():
            with FEATURE_ORDER_PATH.open("r", encoding="utf-8", newline="") as handle:
                values = [row[0].strip() for row in csv.reader(handle) if row]
            return [value for value in values if value and value != "feature"]


        def load_label_names():
            data = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
            return [data[str(index)] for index in sorted((int(key) for key in data.keys()))]


        def softmax(x):
            x = np.asarray(x, dtype=np.float64)
            x = x - np.max(x)
            exps = np.exp(x)
            return exps / np.sum(exps)


        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
        feature_names = load_feature_order()
        label_names = load_label_names()


        def infer(features):
            X = np.asarray(features, dtype=np.float32).reshape(1, -1)
            if X.shape[1] != len(feature_names):
                raise ValueError(f"Se esperaban {{len(feature_names)}} features y llegaron {{X.shape[1]}}")

            if scaler is not None:
                X = scaler.transform(X)

            pred = int(model.predict(X)[0])
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(X)[0]))
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(X)
                confidence = float(np.max(softmax(decision[0] if decision.ndim > 1 else [0.0, decision[0]])))
            else:
                confidence = None

            return pred, confidence


        def on_connect(client, userdata, flags, rc):
            print(f"[RPI] MQTT conectado rc={{rc}} | modelo={display_name}")
            client.subscribe(TOPIC_FEATURES)


        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                client_id = payload.get("client_id", "unknown")
                features = payload["features"]
                pred, confidence = infer(features)

                result = {{
                    "client_id": client_id,
                    "model": "{display_name}",
                    "predicted_class": pred,
                    "predicted_label": label_names[pred],
                    "confidence": confidence,
                }}
                print("[RPI]", json.dumps(result))
                client.publish(TOPIC_PREDICTIONS, json.dumps(result))

                if pred != 0:
                    alert = {{
                        "client_id": client_id,
                        "alert": True,
                        "attack_type": label_names[pred],
                        "attack_probability": confidence,
                        "model": "{display_name}",
                    }}
                    client.publish(TOPIC_ALERTS, json.dumps(alert))
            except Exception as exc:
                print(f"[RPI][ERROR] {{exc}}")


        if __name__ == "__main__":
            client = mqtt.Client(client_id="rpi_gateway_{display_name.lower().replace(' ', '_')}")
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_forever()
        """
    )


def render_raspberry_requirements() -> str:
    return dedent(
        """\
        numpy
        joblib
        scikit-learn
        paho-mqtt
        """
    )


def package_model(spec: dict, artefact_root: Path, bundle_root: Path) -> dict:
    src_dir = artefact_root / spec["artifact_dir"]
    model_path = src_dir / spec["model_file"]
    scaler_path = src_dir / spec["scaler_file"] if spec["scaler_file"] else None

    if not model_path.exists():
        raise FileNotFoundError(f"Falta el modelo entrenado: {model_path}")

    feature_order_path = src_dir / "feature_order.csv"
    label_map_path = src_dir / "label_map.json"
    scaler_params_path = src_dir / "scaler_params.json"

    feature_names = read_feature_order(feature_order_path)
    class_names = read_label_map(label_map_path)

    bundle_dir = bundle_root / spec["bundle_name"]
    esp32_dir = bundle_dir / "esp32"
    raspberry_dir = bundle_dir / "raspberry"
    metadata_dir = bundle_dir / "metadata"

    safe_reset_dir(bundle_dir, bundle_root)
    esp32_dir.mkdir(parents=True, exist_ok=True)
    raspberry_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    model_header_path = export_model(
        str(model_path),
        str(scaler_path) if scaler_path else None,
        str(esp32_dir),
        "model_weights.h",
    )

    header_kb = model_header_path.stat().st_size / 1024.0
    warnings: list[str] = []
    if header_kb > 4096:
        warnings.append(
            "El header C excede 4 MB; esta variante es poco realista para ESP32 y está orientada más a Raspberry/validación."
        )

    copy_if_exists(feature_order_path, metadata_dir / "feature_order.csv")
    copy_if_exists(label_map_path, metadata_dir / "label_map.json")
    copy_if_exists(scaler_params_path, metadata_dir / "scaler_params.json")

    copy_if_exists(feature_order_path, raspberry_dir / "feature_order.csv")
    copy_if_exists(label_map_path, raspberry_dir / "label_map.json")
    copy_if_exists(scaler_params_path, raspberry_dir / "scaler_params.json")
    copy_if_exists(model_path, raspberry_dir / "model.pkl")
    if scaler_path:
        copy_if_exists(scaler_path, raspberry_dir / "scaler.pkl")

    copy_if_exists(feature_order_path, esp32_dir / "feature_order.csv")
    copy_if_exists(label_map_path, esp32_dir / "label_map.json")
    copy_if_exists(scaler_params_path, esp32_dir / "scaler_params.json")

    (esp32_dir / "main_edge_node.cpp").write_text(
        render_esp32_main(class_names, spec["display_name"]),
        encoding="utf-8",
    )
    (raspberry_dir / "predict_local.py").write_text(
        render_raspberry_predict(spec["display_name"]),
        encoding="utf-8",
    )
    (raspberry_dir / "mqtt_gateway.py").write_text(
        render_raspberry_gateway(spec["display_name"]),
        encoding="utf-8",
    )
    (raspberry_dir / "requirements.txt").write_text(
        render_raspberry_requirements(),
        encoding="utf-8",
    )
    (bundle_dir / "README.md").write_text(
        render_bundle_readme(spec["display_name"], class_names, feature_names, header_kb, warnings),
        encoding="utf-8",
    )

    manifest = {
        "bundle": spec["bundle_name"],
        "display_name": spec["display_name"],
        "model_path": str(model_path),
        "scaler_path": str(scaler_path) if scaler_path else None,
        "esp32_header_kb": round(header_kb, 1),
        "warnings": warnings,
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera bundles ESP32/Raspberry por modelo clásico.")
    parser.add_argument("--artefact-root", default=str(DEFAULT_ARTEFACT_ROOT), help="Carpeta con subdirectorios de modelos entrenados.")
    parser.add_argument("--bundle-root", default=str(REPO_ROOT), help="Carpeta donde se crearán los bundles.")
    args = parser.parse_args()

    artefact_root = Path(args.artefact_root).expanduser().resolve()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    manifests = [package_model(spec, artefact_root, bundle_root) for spec in MODEL_SPECS]

    print("\nBundles generados:")
    for manifest in manifests:
        print(f"  - {manifest['bundle']} | header ESP32: {manifest['esp32_header_kb']} KB")
        if manifest["warnings"]:
            for warning in manifest["warnings"]:
                print(f"      WARNING: {warning}")


if __name__ == "__main__":
    main()
