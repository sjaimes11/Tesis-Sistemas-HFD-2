#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

/*
 * Auto-generated from logistic_regression_best.pkl
 * Model: LogisticRegression
 * Classes: normal, mqtt_bruteforce, scan_A
 * Features: 13
 */

#include <math.h>

#define NUM_FEATURES 13

static const float scaler_mean[NUM_FEATURES] = {
    43.79268650f, 3.66618425f, 1.89539289f, 2.65713760f, 13.98416190f, 58.13470509f, 3013.44978490f, 9.14394834f, 0.07881259f, 0.00049752f, 9.13584414f, 51.87352330f, 76.81369928f
};

static const float scaler_scale[NUM_FEATURES] = {
    196.77700981f, 97.04202235f, 33.54320817f, 91.72574463f, 119.55323990f, 10.93673220f, 218919.03787777f, 21.97937194f, 0.28651879f, 0.06254282f, 10.03081721f, 8.56918777f, 29.18449522f
};

void apply_scaler(const float raw[NUM_FEATURES], float scaled[NUM_FEATURES]) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled[i] = (raw[i] - scaler_mean[i]) / scaler_scale[i];
    }
}

// --- Modelo generado por m2cgen ---
#include <string.h>
void score(double * input, double * output) {
    memcpy(output, (double[]){18.917210017530447 + input[0] * 28.722004575325723 + input[1] * -5.635089397934853 + input[2] * 0.897487002689099 + input[3] * 4.506259225304333 + input[4] * 5.242670038539328 + input[5] * -30.588800310639996 + input[6] * -20.890480003250083 + input[7] * 19.65121677211362 + input[8] * -5.469002536407111 + input[9] * -0.21998610420472886 + input[10] * 34.280475192705055 + input[11] * 27.31121860220518 + input[12] * -9.443472192290553, -5.800655514625489 + input[0] * 18.343585221609892 + input[1] * 5.410791013308064 + input[2] * 0.1810858899084515 + input[3] * -1.7052959829407055 + input[4] * -0.3550751660026941 + input[5] * 1.328398946567709 + input[6] * -21.557936266025823 + input[7] * 45.12799373861057 + input[8] * -42.39431396622555 + input[9] * -0.13397515002314206 + input[10] * -9.392894761758205 + input[11] * 2.3263675403269968 + input[12] * -9.105082763784655, -13.11655450287043 + input[0] * -47.065589796932066 + input[1] * 0.22429838462590707 + input[2] * -1.0785728925978224 + input[3] * -2.8009632423621857 + input[4] * -4.887594872536349 + input[5] * 29.260401364071168 + input[6] * 42.448416269277466 + input[7] * -64.77921051072873 + input[8] * 47.8633165026257 + input[9] * 0.3539612542279374 + input[10] * -24.887580430946812 + input[11] * -29.63758614253514 + input[12] * 18.548554956073264}, 3 * sizeof(double));
}



#define NUM_CLASSES 3

int predict_class(const float features[NUM_FEATURES], float* confidence) {
    double input[NUM_FEATURES];
    double scores[NUM_CLASSES];

    for (int i = 0; i < NUM_FEATURES; i++) {
        input[i] = (double)features[i];
    }

    score(input, scores);

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