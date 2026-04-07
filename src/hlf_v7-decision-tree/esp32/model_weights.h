#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

/*
 * Auto-generated from decision_tree_best.pkl
 * Model: DecisionTreeClassifier
 * Classes: normal, mqtt_bruteforce, scan_A
 * Features: 13
 */

#include <math.h>

#define NUM_FEATURES 13

// --- Modelo generado por m2cgen ---
#include <string.h>
void score(double * input, double * output) {
    double var0[3];
    if (input[5] <= 48.11392402648926) {
        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
    } else {
        if (input[10] <= 16.59233283996582) {
            if (input[6] <= 331.0) {
                if (input[4] <= 0.00033104419708251953) {
                    if (input[11] <= 75.5) {
                        if (input[6] <= 73.0) {
                            if (input[5] <= 65.5) {
                                memcpy(var0, (double[]){0.6892177589852009, 0.24904862579281184, 0.061733615221987316}, 3 * sizeof(double));
                            } else {
                                memcpy(var0, (double[]){0.6750764525993884, 0.2591743119266055, 0.06574923547400612}, 3 * sizeof(double));
                            }
                        } else {
                            if (input[12] <= 73.0) {
                                if (input[3] <= 0.00022661685943603516) {
                                    if (input[6] <= 207.0) {
                                        if (input[4] <= 0.0000064373016357421875) {
                                            memcpy(var0, (double[]){0.75, 0.25, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.02857142857142857, 0.9714285714285714, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    }
                                } else {
                                    if (input[1] <= 0.0003135204315185547) {
                                        memcpy(var0, (double[]){0.5476190476190477, 0.40476190476190477, 0.047619047619047616}, 3 * sizeof(double));
                                    } else {
                                        if (input[1] <= 0.00031745433807373047) {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.3333333333333333, 0.6666666666666666, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[3] <= 0.0003135204315185547) {
                                    if (input[2] <= 0.000007331371307373047) {
                                        if (input[6] <= 112.5) {
                                            memcpy(var0, (double[]){0.9169530355097365, 0.021191294387170677, 0.061855670103092786}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8461538461538461, 0.15384615384615385, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                                }
                            }
                        }
                    } else {
                        if (input[11] <= 84.5) {
                            memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                        } else {
                            if (input[6] <= 86.5) {
                                memcpy(var0, (double[]){0.9097408400357462, 0.021447721179624665, 0.06881143878462913}, 3 * sizeof(double));
                            } else {
                                if (input[11] <= 88.5) {
                                    if (input[3] <= 0.00024497509002685547) {
                                        if (input[12] <= 86.5) {
                                            memcpy(var0, (double[]){0.5, 0.5, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[12] <= 86.5) {
                                            memcpy(var0, (double[]){0.9, 0.1, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[6] <= 101.5) {
                                        if (input[6] <= 90.5) {
                                            memcpy(var0, (double[]){0.9127725856697819, 0.0, 0.08722741433021806}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.618421052631579, 0.3256578947368421, 0.05592105263157895}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[11] <= 90.5) {
                                            memcpy(var0, (double[]){0.6666666666666666, 0.0, 0.3333333333333333}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.9848484848484849, 0.015151515151515152}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[10] <= 1.5275251865386963) {
                        if (input[12] <= 69.0) {
                            if (input[1] <= 1.6685180068016052) {
                                if (input[4] <= 5.0018861293792725) {
                                    if (input[1] <= 0.0008634328842163086) {
                                        if (input[1] <= 0.0005384683609008789) {
                                            memcpy(var0, (double[]){0.8390804597701149, 0.08045977011494253, 0.08045977011494253}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[12] <= 63.5) {
                                            memcpy(var0, (double[]){0.7117437722419929, 0.24199288256227758, 0.046263345195729534}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5555555555555556, 0.3888888888888889, 0.05555555555555555}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[4] <= 5.005380868911743) {
                                        if (input[1] <= 1.6676215529441833) {
                                            memcpy(var0, (double[]){0.22727272727272727, 0.7272727272727273, 0.045454545454545456}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.06601466992665037, 0.9266503667481663, 0.007334963325183374}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[3] <= 0.0000054836273193359375) {
                                            memcpy(var0, (double[]){0.4634146341463415, 0.5121951219512195, 0.024390243902439025}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.1724137931034483, 0.8275862068965517, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[2] <= 2.3596922159194946) {
                                    if (input[2] <= 1.3167992234230042) {
                                        if (input[4] <= 4.0960001945495605) {
                                            memcpy(var0, (double[]){0.6875, 0.265625, 0.046875}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.7947368421052632, 0.16052631578947368, 0.04473684210526316}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 1.6685428023338318) {
                                            memcpy(var0, (double[]){0.6904761904761906, 0.14285714285714288, 0.16666666666666669}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6810344827586207, 0.2614942528735632, 0.05747126436781609}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= 2.359983444213867) {
                                        if (input[2] <= 2.35977303981781) {
                                            memcpy(var0, (double[]){0.8606557377049181, 0.0737704918032787, 0.0655737704918033}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9478021978021978, 0.02197802197802198, 0.03021978021978022}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 2.360140800476074) {
                                            memcpy(var0, (double[]){0.5681818181818182, 0.3181818181818182, 0.11363636363636363}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8955223880597015, 0.07462686567164178, 0.029850746268656716}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[12] <= 86.5) {
                                if (input[6] <= 243.0) {
                                    if (input[3] <= 0.001186370849609375) {
                                        if (input[4] <= 0.0003650188446044922) {
                                            memcpy(var0, (double[]){0.6363636363636364, 0.3181818181818182, 0.045454545454545456}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9, 0.0725, 0.0275}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[6] <= 151.0) {
                                            memcpy(var0, (double[]){0.9, 0.03571428571428571, 0.06428571428571428}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5432098765432098, 0.4444444444444444, 0.012345679012345678}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[6] <= 314.0) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[2] <= 2.359676957130432) {
                                            memcpy(var0, (double[]){0.873015873015873, 0.0, 0.12698412698412698}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9477503628447025, 0.0, 0.05224963715529753}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[3] <= 0.0005880594253540039) {
                                    if (input[3] <= 0.0005741119384765625) {
                                        if (input[2] <= 0.00001049041748046875) {
                                            memcpy(var0, (double[]){0.6136363636363636, 0.38636363636363635, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                                    }
                                } else {
                                    if (input[4] <= 0.00074005126953125) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[5] <= 90.5) {
                                            memcpy(var0, (double[]){0.8666666666666667, 0.13333333333333333, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.375, 0.625, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[4] <= 0.0021245479583740234) {
                            if (input[4] <= 0.0004628896713256836) {
                                if (input[1] <= 0.00024110078811645508) {
                                    if (input[3] <= 0.000004887580871582031) {
                                        if (input[3] <= 0.0000029802322387695312) {
                                            memcpy(var0, (double[]){0.8620689655172413, 0.0, 0.13793103448275862}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.892292490118577, 0.0, 0.10770750988142293}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.00021135807037353516) {
                                            memcpy(var0, (double[]){0.8207024029574861, 0.0, 0.17929759704251386}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8746081504702194, 0.0, 0.12539184952978055}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[3] <= 0.00001704692840576172) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[1] <= 0.00025022029876708984) {
                                            memcpy(var0, (double[]){0.8076923076923077, 0.0, 0.19230769230769232}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                memcpy(var0, (double[]){0.9471864088100994, 0.0, 0.052813591189900616}, 3 * sizeof(double));
                            }
                        } else {
                            if (input[10] <= 7.521236181259155) {
                                if (input[4] <= 0.0033255815505981445) {
                                    if (input[2] <= 0.001080302579794079) {
                                        if (input[2] <= 0.0008550786587875336) {
                                            memcpy(var0, (double[]){0.8045977011494253, 0.0, 0.19540229885057472}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9096406972607612, 0.0, 0.0903593027392387}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 0.0010806475766003132) {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8722610722610723, 0.0, 0.12773892773892773}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[3] <= 0.00013899803161621094) {
                                        if (input[4] <= 0.010504961013793945) {
                                            memcpy(var0, (double[]){0.7973962571196095, 0.0, 0.20260374288039057}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    }
                                }
                            } else {
                                if (input[3] <= 1877.335205078125) {
                                    if (input[12] <= 88.5) {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[5] <= 75.25) {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8181818181818182, 0.18181818181818182, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[1] <= 0.875930342823267) {
                    if (input[5] <= 63.214284896850586) {
                        if (input[1] <= 0.00039408603333868086) {
                            if (input[10] <= 16.450876235961914) {
                                memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                            } else {
                                if (input[3] <= 0.0000029802322387695312) {
                                    memcpy(var0, (double[]){0.6666666666666666, 0.0, 0.3333333333333333}, 3 * sizeof(double));
                                } else {
                                    if (input[1] <= 0.00018107891082763672) {
                                        memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                                    } else {
                                        if (input[3] <= 0.0000064373016357421875) {
                                            memcpy(var0, (double[]){0.125, 0.0, 0.875}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[1] <= 0.0005050897598266602) {
                                if (input[2] <= 0.00046374004159588367) {
                                    if (input[1] <= 0.0004199942050036043) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[3] <= 0.000014543533325195312) {
                                            memcpy(var0, (double[]){0.7142857142857143, 0.0, 0.2857142857142857}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= 0.0004583398549584672) {
                                        memcpy(var0, (double[]){0.75, 0.0, 0.25}, 3 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    }
                                }
                            } else {
                                if (input[12] <= 101.0) {
                                    if (input[1] <= 0.0005323290824890137) {
                                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[3] <= 0.000004410743713378906) {
                                            memcpy(var0, (double[]){0.9090909090909091, 0.0, 0.09090909090909091}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.3125, 0.0, 0.6875}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                }
                            }
                        }
                    } else {
                        if (input[11] <= 698.0) {
                            if (input[2] <= 0.0003826239990303293) {
                                if (input[4] <= 0.0010340213775634766) {
                                    if (input[2] <= 0.0002333501324756071) {
                                        if (input[4] <= 0.0005184412002563477) {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.16, 0.0, 0.84}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.0002913276257459074) {
                                            memcpy(var0, (double[]){0.1, 0.0, 0.9}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.29545454545454547, 0.0, 0.7045454545454546}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.75, 0.0, 0.25}, 3 * sizeof(double));
                                }
                            } else {
                                if (input[1] <= 0.0006491740350611508) {
                                    if (input[2] <= 0.00046957822632975876) {
                                        if (input[4] <= 0.0012534856796264648) {
                                            memcpy(var0, (double[]){0.7142857142857143, 0.0, 0.2857142857142857}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.25, 0.0, 0.75}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.0005572239751927555) {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6923076923076923, 0.0, 0.3076923076923077}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[4] <= 0.0026481151580810547) {
                                        if (input[2] <= 0.0009270935843233019) {
                                            memcpy(var0, (double[]){0.1111111111111111, 0.0, 0.8888888888888888}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                        }
                    }
                } else {
                    if (input[2] <= 2.295677661895752) {
                        if (input[1] <= 3.436071038246155) {
                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                        }
                    } else {
                        if (input[1] <= 35.034420013427734) {
                            if (input[9] <= 3.5) {
                                if (input[0] <= 7.5) {
                                    if (input[2] <= 2.3602761030197144) {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.5, 0.5}, 3 * sizeof(double));
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                                }
                            } else {
                                memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                            }
                        } else {
                            if (input[2] <= 288.4137725830078) {
                                if (input[6] <= 568.0) {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){0.5, 0.5, 0.0}, 3 * sizeof(double));
                                }
                            } else {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                            }
                        }
                    }
                }
            }
        } else {
            if (input[2] <= 0.0010146680288016796) {
                if (input[5] <= 63.5) {
                    if (input[4] <= 0.0007520914077758789) {
                        if (input[12] <= 101.5) {
                            if (input[4] <= 0.0007375478744506836) {
                                if (input[3] <= 0.000007987022399902344) {
                                    if (input[10] <= 16.846198081970215) {
                                        if (input[4] <= 0.0006580352783203125) {
                                            memcpy(var0, (double[]){0.5176470588235295, 0.0, 0.4823529411764706}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6956521739130435, 0.0, 0.30434782608695654}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[4] <= 0.0006340742111206055) {
                                            memcpy(var0, (double[]){0.532608695652174, 0.0, 0.4673913043478261}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6891891891891891, 0.0, 0.3108108108108108}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= 0.00017360845959046856) {
                                        if (input[5] <= 63.214284896850586) {
                                            memcpy(var0, (double[]){0.6666666666666666, 0.0, 0.3333333333333333}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8666666666666667, 0.0, 0.13333333333333333}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[3] <= 0.000008463859558105469) {
                                            memcpy(var0, (double[]){0.9333333333333333, 0.0, 0.06666666666666667}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6648531011969532, 0.0, 0.33514689880304677}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[6] <= 442.5) {
                                    memcpy(var0, (double[]){0.6666666666666666, 0.0, 0.3333333333333333}, 3 * sizeof(double));
                                } else {
                                    if (input[3] <= 0.0000045299530029296875) {
                                        if (input[3] <= 0.0000035762786865234375) {
                                            memcpy(var0, (double[]){0.42857142857142855, 0.0, 0.5714285714285714}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[5] <= 63.35714149475098) {
                                            memcpy(var0, (double[]){0.4, 0.0, 0.6}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.4166666666666667, 0.0, 0.5833333333333334}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                        }
                    } else {
                        if (input[3] <= 0.00010156631469726562) {
                            if (input[5] <= 63.35714149475098) {
                                if (input[10] <= 16.87636089324951) {
                                    if (input[4] <= 0.0009340047836303711) {
                                        if (input[2] <= 0.0003826362662948668) {
                                            memcpy(var0, (double[]){0.6847826086956522, 0.0, 0.31521739130434784}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.14285714285714285, 0.0, 0.8571428571428571}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.0007531642913818359) {
                                            memcpy(var0, (double[]){0.8349097162510748, 0.0, 0.1650902837489252}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6433566433566433, 0.0, 0.35664335664335667}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                }
                            } else {
                                if (input[4] <= 0.0023775100708007812) {
                                    if (input[10] <= 17.00676441192627) {
                                        if (input[2] <= 0.000395814873627387) {
                                            memcpy(var0, (double[]){0.7051170858629662, 0.0, 0.2948829141370338}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8131132917038358, 0.0, 0.18688670829616413}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    }
                                } else {
                                    if (input[1] <= 0.0010439157485961914) {
                                        if (input[3] <= 0.000007033348083496094) {
                                            memcpy(var0, (double[]){0.931899641577061, 0.0, 0.06810035842293907}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8024691358024691, 0.0, 0.19753086419753085}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[10] <= 17.00676441192627) {
                                            memcpy(var0, (double[]){0.23809523809523808, 0.0, 0.7619047619047619}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[10] <= 16.972740173339844) {
                                if (input[2] <= 0.00027296532061882317) {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                } else {
                                    if (input[10] <= 16.77214527130127) {
                                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[3] <= 0.00024831295013427734) {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                            }
                        }
                    }
                } else {
                    if (input[10] <= 18.507004737854004) {
                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                    } else {
                        if (input[2] <= 0.00022879424068378285) {
                            if (input[2] <= 0.00013896613381803036) {
                                if (input[12] <= 100.5) {
                                    memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                } else {
                                    if (input[12] <= 162.5) {
                                        if (input[6] <= 532.5) {
                                            memcpy(var0, (double[]){0.8232758620689655, 0.0, 0.17672413793103448}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[10] <= 39.389448165893555) {
                                            memcpy(var0, (double[]){0.4166666666666667, 0.0, 0.5833333333333334}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8181818181818182, 0.0, 0.18181818181818182}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[6] <= 485.5) {
                                    if (input[3] <= 0.000042557716369628906) {
                                        if (input[10] <= 24.522351264953613) {
                                            memcpy(var0, (double[]){0.8810641627543035, 0.0, 0.1189358372456964}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6666666666666666, 0.0, 0.3333333333333333}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[4] <= 0.0007020235061645508) {
                                            memcpy(var0, (double[]){0.7655172413793103, 0.0, 0.23448275862068965}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.25, 0.0, 0.75}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= 0.0002355972901568748) {
                                        if (input[12] <= 155.0) {
                                            memcpy(var0, (double[]){0.9088983050847458, 0.0, 0.09110169491525423}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8101851851851852, 0.0, 0.18981481481481483}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[12] <= 147.5) {
                                            memcpy(var0, (double[]){0.9555555555555556, 0.0, 0.044444444444444446}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8906666666666667, 0.0, 0.10933333333333334}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[4] <= 0.002083420753479004) {
                                if (input[2] <= 0.00030683008662890643) {
                                    if (input[2] <= 0.000306502275634557) {
                                        if (input[10] <= 18.93755340576172) {
                                            memcpy(var0, (double[]){0.3333333333333333, 0.0, 0.6666666666666666}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9333605054004483, 0.0, 0.06663949459955167}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[6] <= 497.0) {
                                            memcpy(var0, (double[]){0.625, 0.0, 0.375}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8888888888888888, 0.0, 0.1111111111111111}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= 0.000664333492750302) {
                                        if (input[5] <= 65.64285659790039) {
                                            memcpy(var0, (double[]){0.8392857142857143, 0.0, 0.16071428571428573}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9622468216754484, 0.0, 0.03775317832455165}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[5] <= 68.92856979370117) {
                                            memcpy(var0, (double[]){0.9067599067599068, 0.0, 0.09324009324009325}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.871517027863777, 0.0, 0.12848297213622292}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[3] <= 0.000006556510925292969) {
                                    if (input[6] <= 512.5) {
                                        if (input[12] <= 117.5) {
                                            memcpy(var0, (double[]){0.9324055666003976, 0.0, 0.06759443339960239}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8778625954198473, 0.0, 0.12213740458015267}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 0.0009693104948382825) {
                                            memcpy(var0, (double[]){0.9930555555555556, 0.0, 0.006944444444444444}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9047619047619048, 0.0, 0.09523809523809523}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[6] <= 485.5) {
                                        if (input[6] <= 472.0) {
                                            memcpy(var0, (double[]){0.8942857142857142, 0.0, 0.10571428571428572}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.7394957983193278, 0.0, 0.2605042016806723}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[4] <= 0.0021185874938964844) {
                                            memcpy(var0, (double[]){0.7666666666666667, 0.0, 0.23333333333333334}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9169611307420494, 0.0, 0.08303886925795052}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[12] <= 101.5) {
                    if (input[10] <= 19.21459674835205) {
                        if (input[4] <= 0.00593256950378418) {
                            if (input[4] <= 0.002698063850402832) {
                                if (input[2] <= 0.0011316498275846243) {
                                    if (input[3] <= 0.00001800060272216797) {
                                        if (input[3] <= 0.0000026226043701171875) {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5945945945945946, 0.0, 0.40540540540540543}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                    }
                                } else {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                }
                            } else {
                                if (input[1] <= 0.0015340645913966) {
                                    if (input[4] <= 0.004473447799682617) {
                                        if (input[1] <= 0.0007034142909105867) {
                                            memcpy(var0, (double[]){0.38461538461538464, 0.0, 0.6153846153846154}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.66, 0.0, 0.34}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[5] <= 63.35714149475098) {
                                            memcpy(var0, (double[]){0.6, 0.0, 0.4}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8444444444444444, 0.0, 0.15555555555555556}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[4] <= 0.005569934844970703) {
                                        if (input[4] <= 0.004320502281188965) {
                                            memcpy(var0, (double[]){0.5625, 0.0, 0.4375}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.23809523809523808, 0.0, 0.7619047619047619}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.8, 0.0, 0.2}, 3 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            if (input[5] <= 64.49999809265137) {
                                if (input[3] <= 0.000012040138244628906) {
                                    if (input[1] <= 0.0015774369239807129) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[4] <= 0.010123014450073242) {
                                            memcpy(var0, (double[]){0.125, 0.0, 0.875}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                }
                            } else {
                                memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                            }
                        }
                    } else {
                        if (input[1] <= 0.0012407501926645637) {
                            if (input[1] <= 0.0008982419967651367) {
                                if (input[3] <= 0.0000029802322387695312) {
                                    memcpy(var0, (double[]){0.75, 0.0, 0.25}, 3 * sizeof(double));
                                } else {
                                    if (input[2] <= 0.0010165093117393553) {
                                        memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                                    } else {
                                        if (input[3] <= 0.000019073486328125) {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8, 0.0, 0.2}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[3] <= 0.0000025033950805664062) {
                                    if (input[6] <= 462.5) {
                                        memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.3333333333333333, 0.0, 0.6666666666666666}, 3 * sizeof(double));
                                    }
                                } else {
                                    if (input[4] <= 0.003993034362792969) {
                                        if (input[2] <= 0.0013505139504559338) {
                                            memcpy(var0, (double[]){0.8545454545454545, 0.0, 0.14545454545454545}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.25, 0.0, 0.75}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            if (input[2] <= 0.0017049199668690562) {
                                if (input[1] <= 0.0013762513408437371) {
                                    if (input[5] <= 66.07143020629883) {
                                        if (input[4] <= 0.0031595230102539062) {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6, 0.0, 0.4}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 0.001398326363414526) {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[4] <= 0.003907561302185059) {
                                        if (input[10] <= 19.6217041015625) {
                                            memcpy(var0, (double[]){0.75, 0.0, 0.25}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                    }
                                }
                            } else {
                                if (input[4] <= 0.0047653913497924805) {
                                    memcpy(var0, (double[]){0.3333333333333333, 0.0, 0.6666666666666666}, 3 * sizeof(double));
                                } else {
                                    if (input[6] <= 463.5) {
                                        if (input[2] <= 0.001826289517339319) {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5714285714285714, 0.0, 0.42857142857142855}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[6] <= 452.0) {
                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                    } else {
                        if (input[1] <= 0.0011925101280212402) {
                            if (input[3] <= 0.000007510185241699219) {
                                if (input[1] <= 0.0010573267936706543) {
                                    if (input[2] <= 0.0014212184469215572) {
                                        if (input[5] <= 75.78571319580078) {
                                            memcpy(var0, (double[]){0.9266917293233082, 0.0, 0.07330827067669173}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9850746268656716, 0.0, 0.014925373134328358}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[5] <= 76.14285659790039) {
                                            memcpy(var0, (double[]){0.8913043478260869, 0.0, 0.10869565217391304}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.5}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[4] <= 0.0024993419647216797) {
                                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                    } else {
                                        if (input[10] <= 19.69076633453369) {
                                            memcpy(var0, (double[]){0.6923076923076923, 0.0, 0.3076923076923077}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8529411764705882, 0.0, 0.14705882352941177}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[3] <= 0.00007748603820800781) {
                                    if (input[6] <= 463.5) {
                                        if (input[1] <= 0.0008014043269213289) {
                                            memcpy(var0, (double[]){0.8888888888888888, 0.0, 0.1111111111111111}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 0.001523671206086874) {
                                            memcpy(var0, (double[]){0.8344155844155844, 0.0, 0.16558441558441558}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[3] <= 0.00010001659393310547) {
                                        memcpy(var0, (double[]){0.25, 0.0, 0.75}, 3 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            if (input[4] <= 0.00795447826385498) {
                                if (input[6] <= 529.5) {
                                    if (input[10] <= 34.83354187011719) {
                                        if (input[5] <= 69.21428680419922) {
                                            memcpy(var0, (double[]){0.5701754385964912, 0.0, 0.4298245614035088}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.7043010752688172, 0.0, 0.2956989247311828}, 3 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                                    }
                                } else {
                                    if (input[3] <= 0.0001214742660522461) {
                                        if (input[1] <= 0.0020905733108520508) {
                                            memcpy(var0, (double[]){0.9555555555555556, 0.0, 0.044444444444444446}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.3333333333333333, 0.0, 0.6666666666666666}, 3 * sizeof(double));
                                        }
                                    } else {
                                        if (input[4] <= 0.0031720399856567383) {
                                            memcpy(var0, (double[]){0.6666666666666666, 0.0, 0.3333333333333333}, 3 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.3333333333333333, 0.0, 0.6666666666666666}, 3 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                memcpy(var0, (double[]){0.2222222222222222, 0.1111111111111111, 0.6666666666666666}, 3 * sizeof(double));
                            }
                        }
                    }
                }
            }
        }
    }
    memcpy(output, var0, 3 * sizeof(double));
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
    return predict_class(raw_features, confidence);
}



#endif // MODEL_WEIGHTS_H