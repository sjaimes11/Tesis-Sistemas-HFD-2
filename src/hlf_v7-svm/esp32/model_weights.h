#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

/*
 * Auto-generated from svm_selected.pkl
 * Model: LinearSVC
 * Classes: normal, mqtt_bruteforce, scan_A
 * Features: 13
 */

#include <math.h>

#define NUM_FEATURES 13

static const float scaler_mean[NUM_FEATURES] = {
    103.79565278f, 2.89939490f, 3.43458862f, 1.53962538f, 27.58548934f, 55.76557892f, 6458.14294444f, 20.87595833f, 0.13356944f, 0.00108333f, 6.89369863f, 51.06931944f, 70.11544444f
};

static const float scaler_scale[NUM_FEATURES] = {
    183.93250900f, 69.35927259f, 24.24101668f, 65.53442968f, 90.84690104f, 12.77188640f, 122741.03020617f, 31.87493686f, 0.37186584f, 0.09203830f, 8.62585243f, 11.01488073f, 27.11647726f
};

void apply_scaler(const float raw[NUM_FEATURES], float scaled[NUM_FEATURES]) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled[i] = (raw[i] - scaler_mean[i]) / scaler_scale[i];
    }
}

// --- Modelo generado por m2cgen ---
#include <string.h>
void score(double * input, double * output) {
    memcpy(output, (double[]){-7.5782380161775595 + input[0] * 11.797080497324158 + input[1] * 13.925720746965748 + input[2] * 31.538290891926746 + input[3] * 31.443470548628525 + input[4] * -61.785073129506515 + input[5] * 33.61768106582074 + input[6] * -10.240831298878195 + input[7] * -1.9650225791634717 + input[8] * 0.000017298721150889414 + input[9] * 0.00811876768080124 + input[10] * 0.49151436910482765 + input[11] * -22.553960969034055 + input[12] * -15.849453435490501, 7.184622394731038 + input[0] * 23.81047778203895 + input[1] * -0.09280828620121381 + input[2] * -0.6569856642942585 + input[3] * -0.8009661748461858 + input[4] * 1.1978291828389889 + input[5] * 5.134729947112952 + input[6] * -12.918665790749117 + input[7] * -2.806028755951467 + input[8] * -2.5806242987460615 + input[9] * -0.08456632262523357 + input[10] * -0.04852929234845194 + input[11] * -1.538522751171746 + input[12] * -5.84956041750662, -23.46468331183989 + input[0] * -63.73740199855933 + input[1] * 1.0179564544856023 + input[2] * -1.342966522936905 + input[3] * -1.736998416264113 + input[4] * 1.9116078038896664 + input[5] * 0.06515628028048259 + input[6] * 34.59111093552444 + input[7] * 10.409943150460103 + input[8] * 8.428213563644963 + input[9] * 0.27619015589529256 + input[10] * -0.06676627835815337 + input[11] * -1.44997172448037 + input[12] * 0.03377929985046594}, 3 * sizeof(double));
}



#define NUM_CLASSES 3

int predict_class(const float features[NUM_FEATURES], float* confidence) {
    double scores[NUM_CLASSES];
    score(features, scores);

    int best = 0;
    double best_score = scores[0];
    double sum_exp = 0.0;

    for (int i = 1; i < NUM_CLASSES; i++) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best = i;
        }
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        sum_exp += exp(scores[i] - best_score);
    }
    *confidence = (float)(1.0 / sum_exp);

    return best;
}


int classify(const float raw_features[NUM_FEATURES], float* confidence) {
    float scaled[NUM_FEATURES];
    apply_scaler(raw_features, scaled);
    return predict_class(scaled, confidence);
}



#endif // MODEL_WEIGHTS_H