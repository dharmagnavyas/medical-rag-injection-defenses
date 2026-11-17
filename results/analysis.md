# Analysis of 21,600 scored runs

Models: llama3, mistral, qwen, gemma, meditron, biomistral. Conditions: baseline, sanitize, spotlight, struq, mapfilter. ASR rows use the `white_text` technique (the one every condition ran); paired tests use the MPIB subset. CIs are 95% Wilson intervals; bootstrap CIs use 2000 resamples, seed 42. Figures: results/figures/.


### T1. ASR, FRR and utility per model and condition

ASR: attack items, `white_text` rows. FRR: genuine items refused. Utility: MedQA answer correct.

| model | condition | n_attack | ASR % | ASR MPIB % | n_genuine | FRR % | utility % | no_answer % |
|---|---|---|---|---|---|---|---|---|
| llama3 | baseline | 300 | 42.0 [36.6, 47.7] | 84.0 [77.3, 89.0] | 300 | 0.0 [0.0, 1.3] | 51.3 [45.7, 56.9] | 0.0 |
| llama3 | sanitize | 300 | 11.0 [7.9, 15.0] | 22.0 [16.1, 29.3] | 300 | 0.0 [0.0, 1.3] | 51.7 [46.0, 57.3] | 0.0 |
| llama3 | spotlight | 300 | 40.3 [34.9, 46.0] | 80.7 [73.6, 86.2] | 300 | 0.0 [0.0, 1.3] | 50.0 [44.4, 55.6] | 0.0 |
| llama3 | struq | 300 | 41.7 [36.2, 47.3] | 82.7 [75.8, 87.9] | 300 | 0.0 [0.0, 1.3] | 52.0 [46.4, 57.6] | 0.3 |
| llama3 | mapfilter | 300 | 39.3 [34.0, 45.0] | 78.7 [71.4, 84.5] | 300 | 0.0 [0.0, 1.3] | 55.0 [49.3, 60.5] | 0.0 |
| mistral | baseline | 300 | 40.3 [34.9, 46.0] | 80.7 [73.6, 86.2] | 300 | 0.0 [0.0, 1.3] | 47.3 [41.8, 53.0] | 1.0 |
| mistral | sanitize | 300 | 7.7 [5.2, 11.2] | 15.3 [10.4, 22.0] | 300 | 0.0 [0.0, 1.3] | 46.3 [40.8, 52.0] | 0.7 |
| mistral | spotlight | 300 | 40.0 [34.6, 45.6] | 80.0 [72.9, 85.6] | 300 | 0.0 [0.0, 1.3] | 44.3 [38.8, 50.0] | 0.0 |
| mistral | struq | 300 | 41.3 [35.9, 47.0] | 82.7 [75.8, 87.9] | 300 | 0.0 [0.0, 1.3] | 44.7 [39.1, 50.3] | 1.3 |
| mistral | mapfilter | 300 | 38.3 [33.0, 43.9] | 76.7 [69.3, 82.7] | 300 | 0.3 [0.1, 1.9] | 47.7 [42.1, 53.3] | 1.0 |
| qwen | baseline | 300 | 44.7 [39.1, 50.3] | 89.3 [83.4, 93.3] | 300 | 0.0 [0.0, 1.3] | 55.0 [49.3, 60.5] | 0.0 |
| qwen | sanitize | 300 | 16.7 [12.9, 21.3] | 33.3 [26.3, 41.2] | 300 | 0.0 [0.0, 1.3] | 56.7 [51.0, 62.2] | 0.0 |
| qwen | spotlight | 300 | 44.7 [39.1, 50.3] | 89.3 [83.4, 93.3] | 300 | 0.0 [0.0, 1.3] | 53.7 [48.0, 59.2] | 0.0 |
| qwen | struq | 300 | 45.3 [39.8, 51.0] | 90.7 [84.9, 94.4] | 300 | 0.0 [0.0, 1.3] | 55.0 [49.3, 60.5] | 0.0 |
| qwen | mapfilter | 300 | 42.3 [36.9, 48.0] | 84.7 [78.0, 89.6] | 300 | 0.0 [0.0, 1.3] | 55.7 [50.0, 61.2] | 0.0 |
| gemma | baseline | 300 | 34.0 [28.9, 39.5] | 68.0 [60.2, 74.9] | 300 | 0.0 [0.0, 1.3] | 46.0 [40.4, 51.7] | 0.0 |
| gemma | sanitize | 300 | 7.7 [5.2, 11.2] | 15.3 [10.4, 22.0] | 300 | 0.0 [0.0, 1.3] | 45.7 [40.1, 51.3] | 0.0 |
| gemma | spotlight | 300 | 34.7 [29.5, 40.2] | 69.3 [61.5, 76.2] | 300 | 0.3 [0.1, 1.9] | 43.7 [38.2, 49.3] | 0.3 |
| gemma | struq | 300 | 34.3 [29.2, 39.9] | 68.7 [60.9, 75.5] | 300 | 0.3 [0.1, 1.9] | 41.3 [35.9, 47.0] | 0.3 |
| gemma | mapfilter | 300 | 33.7 [28.6, 39.2] | 67.3 [59.5, 74.3] | 300 | 0.3 [0.1, 1.9] | 46.0 [40.4, 51.7] | 0.3 |
| meditron | baseline | 300 | 36.0 [30.8, 41.6] | 71.3 [63.6, 78.0] | 300 | 0.0 [0.0, 1.3] | 28.7 [23.8, 34.0] | 1.0 |
| meditron | sanitize | 300 | 6.0 [3.8, 9.3] | 11.3 [7.2, 17.4] | 300 | 0.0 [0.0, 1.3] | 28.7 [23.8, 34.0] | 1.3 |
| meditron | spotlight | 300 | 34.7 [29.5, 40.2] | 69.3 [61.5, 76.2] | 300 | 0.0 [0.0, 1.3] | 28.3 [23.5, 33.7] | 0.7 |
| meditron | struq | 300 | 41.7 [36.2, 47.3] | 66.0 [58.1, 73.1] | 300 | 0.0 [0.0, 1.3] | 30.3 [25.4, 35.8] | 0.7 |
| meditron | mapfilter | 300 | 30.3 [25.4, 35.8] | 60.7 [52.7, 68.1] | 300 | 0.0 [0.0, 1.3] | 30.7 [25.7, 36.1] | 2.0 |
| biomistral | baseline | 300 | 38.0 [32.7, 43.6] | 73.3 [65.7, 79.8] | 300 | 0.0 [0.0, 1.3] | 40.7 [35.3, 46.3] | 10.0 |
| biomistral | sanitize | 300 | 9.0 [6.3, 12.8] | 18.0 [12.7, 24.9] | 300 | 0.0 [0.0, 1.3] | 38.0 [32.7, 43.6] | 7.3 |
| biomistral | spotlight | 300 | 37.3 [32.1, 42.9] | 73.3 [65.7, 79.8] | 300 | 0.0 [0.0, 1.3] | 36.3 [31.1, 41.9] | 12.3 |
| biomistral | struq | 300 | 35.0 [29.8, 40.6] | 70.0 [62.2, 76.8] | 300 | 0.0 [0.0, 1.3] | 40.7 [35.3, 46.3] | 13.3 |
| biomistral | mapfilter | 300 | 34.7 [29.5, 40.2] | 69.3 [61.5, 76.2] | 300 | 0.0 [0.0, 1.3] | 39.7 [34.3, 45.3] | 9.3 |

### T2. ASR by attack source

| model | condition | MPIB ASR % | AdvBench ASR % |
|---|---|---|---|
| llama3 | baseline | 84.0 [77.3, 89.0] | 0.0 [0.0, 2.5] |
| llama3 | sanitize | 22.0 [16.1, 29.3] | 0.0 [0.0, 2.5] |
| llama3 | spotlight | 80.7 [73.6, 86.2] | 0.0 [0.0, 2.5] |
| llama3 | struq | 82.7 [75.8, 87.9] | 0.7 [0.1, 3.7] |
| llama3 | mapfilter | 78.7 [71.4, 84.5] | 0.0 [0.0, 2.5] |
| mistral | baseline | 80.7 [73.6, 86.2] | 0.0 [0.0, 2.5] |
| mistral | sanitize | 15.3 [10.4, 22.0] | 0.0 [0.0, 2.5] |
| mistral | spotlight | 80.0 [72.9, 85.6] | 0.0 [0.0, 2.5] |
| mistral | struq | 82.7 [75.8, 87.9] | 0.0 [0.0, 2.5] |
| mistral | mapfilter | 76.7 [69.3, 82.7] | 0.0 [0.0, 2.5] |
| qwen | baseline | 89.3 [83.4, 93.3] | 0.0 [0.0, 2.5] |
| qwen | sanitize | 33.3 [26.3, 41.2] | 0.0 [0.0, 2.5] |
| qwen | spotlight | 89.3 [83.4, 93.3] | 0.0 [0.0, 2.5] |
| qwen | struq | 90.7 [84.9, 94.4] | 0.0 [0.0, 2.5] |
| qwen | mapfilter | 84.7 [78.0, 89.6] | 0.0 [0.0, 2.5] |
| gemma | baseline | 68.0 [60.2, 74.9] | 0.0 [0.0, 2.5] |
| gemma | sanitize | 15.3 [10.4, 22.0] | 0.0 [0.0, 2.5] |
| gemma | spotlight | 69.3 [61.5, 76.2] | 0.0 [0.0, 2.5] |
| gemma | struq | 68.7 [60.9, 75.5] | 0.0 [0.0, 2.5] |
| gemma | mapfilter | 67.3 [59.5, 74.3] | 0.0 [0.0, 2.5] |
| meditron | baseline | 71.3 [63.6, 78.0] | 0.7 [0.1, 3.7] |
| meditron | sanitize | 11.3 [7.2, 17.4] | 0.7 [0.1, 3.7] |
| meditron | spotlight | 69.3 [61.5, 76.2] | 0.0 [0.0, 2.5] |
| meditron | struq | 66.0 [58.1, 73.1] | 17.3 [12.1, 24.2] |
| meditron | mapfilter | 60.7 [52.7, 68.1] | 0.0 [0.0, 2.5] |
| biomistral | baseline | 73.3 [65.7, 79.8] | 2.7 [1.0, 6.7] |
| biomistral | sanitize | 18.0 [12.7, 24.9] | 0.0 [0.0, 2.5] |
| biomistral | spotlight | 73.3 [65.7, 79.8] | 1.3 [0.4, 4.7] |
| biomistral | struq | 70.0 [62.2, 76.8] | 0.0 [0.0, 2.5] |
| biomistral | mapfilter | 69.3 [61.5, 76.2] | 0.0 [0.0, 2.5] |

### T3. Sanitization: ASR by hiding technique

| model | metadata % | tiny_font % | white_text % |
|---|---|---|---|
| llama3 | 11.3 [8.2, 15.4] | 11.3 [8.2, 15.4] | 11.0 [7.9, 15.0] |
| mistral | 8.3 [5.7, 12.0] | 8.7 [6.0, 12.4] | 7.7 [5.2, 11.2] |
| qwen | 16.7 [12.9, 21.3] | 16.7 [12.9, 21.3] | 16.7 [12.9, 21.3] |
| gemma | 7.3 [4.9, 10.9] | 7.3 [4.9, 10.9] | 7.7 [5.2, 11.2] |
| meditron | 8.0 [5.4, 11.6] | 6.7 [4.4, 10.1] | 6.0 [3.8, 9.3] |
| biomistral | 10.0 [7.1, 13.9] | 9.7 [6.8, 13.5] | 9.0 [6.3, 12.8] |

### T4. Exact McNemar: each defense vs no defense (paired by item)

Holm correction within each model and subset (4 tests). b and c are the discordant counts.

| subset | model | defense | n_items | baseline->safe (b) | safe->attacked (c) | delta pp | p (exact McNemar) | p (Holm) | significant |
|---|---|---|---|---|---|---|---|---|---|
| all attacks | llama3 | sanitize | 300 | 95 | 2 | -31 | 6e-26 | 2.4e-25 | True |
| all attacks | llama3 | spotlight | 300 | 9 | 4 | -1.7 | 0.2668 | 0.5337 | False |
| all attacks | llama3 | struq | 300 | 6 | 5 | -0.3 | 1 | 1 | False |
| all attacks | llama3 | mapfilter | 300 | 13 | 5 | -2.7 | 0.09625 | 0.2888 | False |
| all attacks | mistral | sanitize | 300 | 99 | 1 | -32.7 | 1.593e-28 | 6.374e-28 | True |
| all attacks | mistral | spotlight | 300 | 5 | 4 | -0.3 | 1 | 1 | False |
| all attacks | mistral | struq | 300 | 4 | 7 | 1 | 0.5488 | 1 | False |
| all attacks | mistral | mapfilter | 300 | 9 | 3 | -2 | 0.146 | 0.438 | False |
| all attacks | qwen | sanitize | 300 | 86 | 2 | -28 | 2.531e-23 | 1.013e-22 | True |
| all attacks | qwen | spotlight | 300 | 2 | 2 | 0 | 1 | 1 | False |
| all attacks | qwen | struq | 300 | 2 | 4 | 0.7 | 0.6875 | 1 | False |
| all attacks | qwen | mapfilter | 300 | 9 | 2 | -2.3 | 0.06543 | 0.1963 | False |
| all attacks | gemma | sanitize | 300 | 81 | 2 | -26.3 | 7.211e-22 | 2.884e-21 | True |
| all attacks | gemma | spotlight | 300 | 7 | 9 | 0.7 | 0.8036 | 1 | False |
| all attacks | gemma | struq | 300 | 6 | 7 | 0.3 | 1 | 1 | False |
| all attacks | gemma | mapfilter | 300 | 8 | 7 | -0.3 | 1 | 1 | False |
| all attacks | meditron | sanitize | 300 | 91 | 1 | -30 | 3.756e-26 | 1.502e-25 | True |
| all attacks | meditron | spotlight | 300 | 20 | 16 | -1.3 | 0.6177 | 0.6177 | False |
| all attacks | meditron | struq | 300 | 20 | 37 | 5.7 | 0.03314 | 0.06629 | False |
| all attacks | meditron | mapfilter | 300 | 29 | 12 | -5.7 | 0.01151 | 0.03452 | True |
| all attacks | biomistral | sanitize | 300 | 88 | 1 | -29 | 2.908e-25 | 1.163e-24 | True |
| all attacks | biomistral | spotlight | 300 | 16 | 14 | -0.7 | 0.8555 | 0.8555 | False |
| all attacks | biomistral | struq | 300 | 13 | 4 | -3 | 0.04904 | 0.1471 | False |
| all attacks | biomistral | mapfilter | 300 | 16 | 6 | -3.3 | 0.05248 | 0.1471 | False |
| MPIB | llama3 | sanitize | 150 | 95 | 2 | -62 | 6e-26 | 2.4e-25 | True |
| MPIB | llama3 | spotlight | 150 | 9 | 4 | -3.3 | 0.2668 | 0.5337 | False |
| MPIB | llama3 | struq | 150 | 6 | 4 | -1.3 | 0.7539 | 0.7539 | False |
| MPIB | llama3 | mapfilter | 150 | 13 | 5 | -5.3 | 0.09625 | 0.2888 | False |
| MPIB | mistral | sanitize | 150 | 99 | 1 | -65.3 | 1.593e-28 | 6.374e-28 | True |
| MPIB | mistral | spotlight | 150 | 5 | 4 | -0.7 | 1 | 1 | False |
| MPIB | mistral | struq | 150 | 4 | 7 | 2 | 0.5488 | 1 | False |
| MPIB | mistral | mapfilter | 150 | 9 | 3 | -4 | 0.146 | 0.438 | False |
| MPIB | qwen | sanitize | 150 | 86 | 2 | -56 | 2.531e-23 | 1.013e-22 | True |
| MPIB | qwen | spotlight | 150 | 2 | 2 | 0 | 1 | 1 | False |
| MPIB | qwen | struq | 150 | 2 | 4 | 1.3 | 0.6875 | 1 | False |
| MPIB | qwen | mapfilter | 150 | 9 | 2 | -4.7 | 0.06543 | 0.1963 | False |
| MPIB | gemma | sanitize | 150 | 81 | 2 | -52.7 | 7.211e-22 | 2.884e-21 | True |
| MPIB | gemma | spotlight | 150 | 7 | 9 | 1.3 | 0.8036 | 1 | False |
| MPIB | gemma | struq | 150 | 6 | 7 | 0.7 | 1 | 1 | False |
| MPIB | gemma | mapfilter | 150 | 8 | 7 | -0.7 | 1 | 1 | False |
| MPIB | meditron | sanitize | 150 | 91 | 1 | -60 | 3.756e-26 | 1.502e-25 | True |
| MPIB | meditron | spotlight | 150 | 19 | 16 | -2 | 0.7359 | 0.7359 | False |
| MPIB | meditron | struq | 150 | 19 | 11 | -5.3 | 0.2005 | 0.401 | False |
| MPIB | meditron | mapfilter | 150 | 28 | 12 | -10.7 | 0.01659 | 0.04977 | True |
| MPIB | biomistral | sanitize | 150 | 84 | 1 | -55.3 | 4.446e-24 | 1.778e-23 | True |
| MPIB | biomistral | spotlight | 150 | 12 | 12 | 0 | 1 | 1 | False |
| MPIB | biomistral | struq | 150 | 9 | 4 | -3.3 | 0.2668 | 0.7137 | False |
| MPIB | biomistral | mapfilter | 150 | 12 | 6 | -4 | 0.2379 | 0.7137 | False |

### T4b. Cochran's Q across the five conditions

| subset | model | n_items | Cochran Q | df | p |
|---|---|---|---|---|---|
| all attacks | llama3 | 300 | 269.8 | 4 | 3.452e-57 |
| all attacks | mistral | 300 | 320.7 | 4 | 3.651e-68 |
| all attacks | qwen | 300 | 274.9 | 4 | 2.85e-58 |
| all attacks | gemma | 300 | 232.2 | 4 | 4.513e-49 |
| all attacks | meditron | 300 | 203.7 | 4 | 5.929e-43 |
| all attacks | biomistral | 300 | 221.3 | 4 | 9.974e-47 |
| MPIB | llama3 | 150 | 270.4 | 4 | 2.576e-57 |
| MPIB | mistral | 150 | 320.7 | 4 | 3.651e-68 |
| MPIB | qwen | 150 | 274.9 | 4 | 2.85e-58 |
| MPIB | gemma | 150 | 232.2 | 4 | 4.513e-49 |
| MPIB | meditron | 150 | 199.7 | 4 | 4.378e-42 |
| MPIB | biomistral | 150 | 222.4 | 4 | 5.602e-47 |

### T5. Change in ASR vs no defense, paired bootstrap 95% CI (percentage points)

| subset | model | defense | n_items | baseline ASR % | defense ASR % | delta pp | CI low | CI high |
|---|---|---|---|---|---|---|---|---|
| all attacks | llama3 | sanitize | 300 | 42.0 | 11.0 | -31.0 | -36.3 | -25.3 |
| all attacks | llama3 | spotlight | 300 | 42.0 | 40.3 | -1.7 | -4.0 | 0.7 |
| all attacks | llama3 | struq | 300 | 42.0 | 41.7 | -0.3 | -2.7 | 1.7 |
| all attacks | llama3 | mapfilter | 300 | 42.0 | 39.3 | -2.7 | -5.3 | 0.0 |
| all attacks | mistral | sanitize | 300 | 40.3 | 7.7 | -32.7 | -38.0 | -27.3 |
| all attacks | mistral | spotlight | 300 | 40.3 | 40.0 | -0.3 | -2.3 | 1.7 |
| all attacks | mistral | struq | 300 | 40.3 | 41.3 | 1.0 | -1.0 | 3.3 |
| all attacks | mistral | mapfilter | 300 | 40.3 | 38.3 | -2.0 | -4.3 | 0.3 |
| all attacks | qwen | sanitize | 300 | 44.7 | 16.7 | -28.0 | -33.3 | -22.7 |
| all attacks | qwen | spotlight | 300 | 44.7 | 44.7 | 0.0 | -1.3 | 1.3 |
| all attacks | qwen | struq | 300 | 44.7 | 45.3 | 0.7 | -1.0 | 2.3 |
| all attacks | qwen | mapfilter | 300 | 44.7 | 42.3 | -2.3 | -4.7 | -0.3 |
| all attacks | gemma | sanitize | 300 | 34.0 | 7.7 | -26.3 | -31.7 | -21.0 |
| all attacks | gemma | spotlight | 300 | 34.0 | 34.7 | 0.7 | -2.0 | 3.3 |
| all attacks | gemma | struq | 300 | 34.0 | 34.3 | 0.3 | -2.0 | 2.7 |
| all attacks | gemma | mapfilter | 300 | 34.0 | 33.7 | -0.3 | -2.7 | 2.3 |
| all attacks | meditron | sanitize | 300 | 36.0 | 6.0 | -30.0 | -35.3 | -25.0 |
| all attacks | meditron | spotlight | 300 | 36.0 | 34.7 | -1.3 | -5.3 | 2.7 |
| all attacks | meditron | struq | 300 | 36.0 | 41.7 | 5.7 | 0.7 | 10.7 |
| all attacks | meditron | mapfilter | 300 | 36.0 | 30.3 | -5.7 | -10.0 | -1.7 |
| all attacks | biomistral | sanitize | 300 | 38.0 | 9.0 | -29.0 | -34.3 | -23.7 |
| all attacks | biomistral | spotlight | 300 | 38.0 | 37.3 | -0.7 | -4.3 | 3.0 |
| all attacks | biomistral | struq | 300 | 38.0 | 35.0 | -3.0 | -5.7 | -0.3 |
| all attacks | biomistral | mapfilter | 300 | 38.0 | 34.7 | -3.3 | -6.3 | -0.3 |
| MPIB | llama3 | sanitize | 150 | 84.0 | 22.0 | -62.0 | -70.0 | -54.0 |
| MPIB | llama3 | spotlight | 150 | 84.0 | 80.7 | -3.3 | -8.0 | 1.3 |
| MPIB | llama3 | struq | 150 | 84.0 | 82.7 | -1.3 | -6.0 | 2.7 |
| MPIB | llama3 | mapfilter | 150 | 84.0 | 78.7 | -5.3 | -10.7 | 0.0 |
| MPIB | mistral | sanitize | 150 | 80.7 | 15.3 | -65.3 | -73.3 | -57.3 |
| MPIB | mistral | spotlight | 150 | 80.7 | 80.0 | -0.7 | -4.7 | 3.3 |
| MPIB | mistral | struq | 150 | 80.7 | 82.7 | 2.0 | -2.0 | 6.0 |
| MPIB | mistral | mapfilter | 150 | 80.7 | 76.7 | -4.0 | -8.7 | 0.0 |
| MPIB | qwen | sanitize | 150 | 89.3 | 33.3 | -56.0 | -64.7 | -47.3 |
| MPIB | qwen | spotlight | 150 | 89.3 | 89.3 | 0.0 | -2.7 | 2.7 |
| MPIB | qwen | struq | 150 | 89.3 | 90.7 | 1.3 | -2.0 | 4.7 |
| MPIB | qwen | mapfilter | 150 | 89.3 | 84.7 | -4.7 | -8.7 | -0.7 |
| MPIB | gemma | sanitize | 150 | 68.0 | 15.3 | -52.7 | -60.7 | -44.7 |
| MPIB | gemma | spotlight | 150 | 68.0 | 69.3 | 1.3 | -4.0 | 6.7 |
| MPIB | gemma | struq | 150 | 68.0 | 68.7 | 0.7 | -4.0 | 5.3 |
| MPIB | gemma | mapfilter | 150 | 68.0 | 67.3 | -0.7 | -6.0 | 4.7 |
| MPIB | meditron | sanitize | 150 | 71.3 | 11.3 | -60.0 | -68.7 | -52.0 |
| MPIB | meditron | spotlight | 150 | 71.3 | 69.3 | -2.0 | -9.3 | 6.0 |
| MPIB | meditron | struq | 150 | 71.3 | 66.0 | -5.3 | -12.7 | 2.0 |
| MPIB | meditron | mapfilter | 150 | 71.3 | 60.7 | -10.7 | -19.3 | -3.3 |
| MPIB | biomistral | sanitize | 150 | 73.3 | 18.0 | -55.3 | -63.3 | -47.3 |
| MPIB | biomistral | spotlight | 150 | 73.3 | 73.3 | 0.0 | -6.7 | 6.0 |
| MPIB | biomistral | struq | 150 | 73.3 | 70.0 | -3.3 | -8.0 | 1.3 |
| MPIB | biomistral | mapfilter | 150 | 73.3 | 69.3 | -4.0 | -10.0 | 1.3 |

### T6. Medical fine-tuning condition: medical model vs general comparator, no-defense prompt

The clean pair is BioMistral vs Mistral (same base). Meditron is Llama-2-based; Llama 3 is the nearest general model run.

| metric | general model | general % | medical model | medical % | delta pp | n_items | p (exact McNemar) |
|---|---|---|---|---|---|---|---|
| ASR (white_text) | mistral | 40.3 | biomistral | 38 | -2.3 | 300 | 0.281 |
| ASR MPIB | mistral | 80.7 | biomistral | 73.3 | -7.3 | 150 | 0.05224 |
| utility | mistral | 47.3 | biomistral | 40.7 | -6.7 | 300 | 0.04221 |
| FRR | mistral | 0 | biomistral | 0 | 0 | 300 | 1 |
| ASR (white_text) | llama3 | 42 | meditron | 36 | -6 | 300 | 0.002935 |
| ASR MPIB | llama3 | 84 | meditron | 71.3 | -12.7 | 150 | 0.001319 |
| utility | llama3 | 51.3 | meditron | 28.7 | -22.7 | 300 | 3.275e-08 |
| FRR | llama3 | 0 | meditron | 0 | 0 | 300 | 1 |

### T7. Genuine questions by MedQA category and condition, pooled over models (RQ3)

| category | n_items | baseline FRR % | baseline no-answer % | baseline utility % | sanitize FRR % | sanitize no-answer % | sanitize utility % | spotlight FRR % | spotlight no-answer % | spotlight utility % | struq FRR % | struq no-answer % | struq utility % | mapfilter FRR % | mapfilter no-answer % | mapfilter utility % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| basic_science | 17 | 0.0 | 2.9 | 33.3 | 0.0 | 2.0 | 31.4 | 1.0 | 2.9 | 33.3 | 1.0 | 2.9 | 35.3 | 1.0 | 2.9 | 33.3 |
| biostat_ethics | 53 | 0.0 | 2.2 | 45.0 | 0.0 | 1.3 | 45.0 | 0.0 | 4.1 | 44.7 | 0.0 | 3.1 | 43.4 | 0.3 | 2.8 | 46.2 |
| diagnosis | 36 | 0.0 | 1.9 | 45.4 | 0.0 | 1.9 | 46.3 | 0.0 | 0.9 | 44.4 | 0.0 | 2.3 | 46.3 | 0.0 | 1.9 | 44.4 |
| findings | 12 | 0.0 | 5.6 | 41.7 | 0.0 | 5.6 | 36.1 | 0.0 | 2.8 | 44.4 | 0.0 | 4.2 | 40.3 | 0.0 | 4.2 | 43.1 |
| investigation | 9 | 0.0 | 0.0 | 51.9 | 0.0 | 0.0 | 50.0 | 0.0 | 3.7 | 42.6 | 0.0 | 0.0 | 46.3 | 0.0 | 0.0 | 50.0 |
| management | 73 | 0.0 | 0.9 | 54.1 | 0.0 | 0.7 | 53.2 | 0.0 | 0.9 | 46.6 | 0.0 | 1.8 | 50.5 | 0.0 | 1.1 | 53.4 |
| other | 8 | 0.0 | 6.2 | 35.4 | 0.0 | 2.1 | 35.4 | 0.0 | 6.2 | 33.3 | 0.0 | 10.4 | 39.6 | 0.0 | 8.3 | 37.5 |
| pathophysiology | 23 | 0.0 | 2.2 | 37.7 | 0.0 | 2.9 | 37.0 | 0.0 | 1.4 | 40.6 | 0.0 | 2.2 | 39.1 | 0.0 | 1.4 | 41.3 |
| pharmacology | 59 | 0.0 | 2.3 | 43.8 | 0.0 | 1.7 | 44.9 | 0.0 | 2.3 | 42.9 | 0.0 | 2.0 | 44.4 | 0.0 | 2.0 | 47.2 |
| risk_prognosis | 10 | 0.0 | 0.0 | 21.7 | 0.0 | 0.0 | 21.7 | 0.0 | 1.7 | 23.3 | 0.0 | 6.7 | 21.7 | 0.0 | 1.7 | 21.7 |

### T8. ASR conditional on the poisoned chunk being in the top-k

| condition | poisoned chunk retrieved | n | ASR % |
|---|---|---|---|
| baseline | True | 1710 | 40.8 [38.5, 43.1] |
| baseline | False | 90 | 8.9 [4.6, 16.6] |
| sanitize | True | 90 | 33.3 [24.5, 43.6] |
| sanitize | False | 1710 | 8.4 [7.2, 9.8] |
| spotlight | True | 1710 | 40.1 [37.8, 42.4] |
| spotlight | False | 90 | 11.1 [6.1, 19.3] |
| struq | True | 1710 | 41.5 [39.1, 43.8] |
| struq | False | 90 | 10.0 [5.4, 17.9] |
| mapfilter | True | 1710 | 38.0 [35.7, 40.3] |
| mapfilter | False | 90 | 6.7 [3.1, 13.8] |

### T9. Scorer and judge diagnostics

| statistic | value |
|---|---|
| judge calibration rows (qwen2.5:7b vs mistral:7b) | 600 |
| judge agreement | 0.917 |
| judge Cohen's kappa | 0.822 |
| judge confusion (primary,second): 00/01/10/11 | 355/46/4/195 |
| judge decisions by qwen2.5:7b | 8146 |
| judge decisions by letter_only | 3329 |
| judge decisions by mistral:7b | 1050 |
| judge decisions by empty | 75 |
| rule_hit & judge agreement on attack rows | 0.839 |
| genuine letters from option text (fallback) | 218 |
| llama3: doc_echo % (attack rows) | 0.000 |
| llama3: mean latency s | 1.500 |
| mistral: doc_echo % (attack rows) | 0.200 |
| mistral: mean latency s | 1.990 |
| qwen: doc_echo % (attack rows) | 0.000 |
| qwen: mean latency s | 1.380 |
| gemma: doc_echo % (attack rows) | 0.000 |
| gemma: mean latency s | 1.350 |
| meditron: doc_echo % (attack rows) | 11.900 |
| meditron: mean latency s | 11.900 |
| biomistral: doc_echo % (attack rows) | 0.000 |
| biomistral: mean latency s | 2.580 |

### T10. Independent review

No review labels found.

### T11. ROUGE-L F1 of the model output against the injected instruction

A graded companion to the binary ASR: how much of the planted text the model reproduces, averaged over attack rows (white-text injection). BLEU/ROUGE are not used for utility because MedQA is multiple choice.

| model | baseline mean | baseline MPIB mean | sanitize mean | sanitize MPIB mean | spotlight mean | spotlight MPIB mean | struq mean | struq MPIB mean | mapfilter mean | mapfilter MPIB mean |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3 | 0.209 | 0.414 | 0.083 | 0.161 | 0.186 | 0.367 | 0.194 | 0.382 | 0.181 | 0.357 |
| mistral | 0.194 | 0.371 | 0.083 | 0.150 | 0.197 | 0.382 | 0.249 | 0.473 | 0.181 | 0.343 |
| qwen | 0.227 | 0.449 | 0.083 | 0.162 | 0.229 | 0.454 | 0.234 | 0.464 | 0.206 | 0.408 |
| gemma | 0.170 | 0.335 | 0.061 | 0.119 | 0.172 | 0.341 | 0.161 | 0.317 | 0.162 | 0.321 |
| meditron | 0.155 | 0.265 | 0.061 | 0.079 | 0.130 | 0.219 | 0.135 | 0.202 | 0.121 | 0.199 |
| biomistral | 0.231 | 0.403 | 0.094 | 0.152 | 0.235 | 0.418 | 0.222 | 0.384 | 0.204 | 0.369 |

### Figures

- README.md
- fig_3_1_pipeline.png
- fig_3_2_hidden_text.png
- fig_3_3_defense_positions.png
- fig_4_1_categories.png
- fig_4_2_attack_types.png
- fig_4_3_lengths.png
- fig_4_4_retrieval.png
- fig_asr_all.png
- fig_asr_by_source.png
- fig_asr_heatmap.png
- fig_asr_mpib.png
- fig_rouge_l_injection.png
- fig_sanitize_by_technique.png
- fig_tradeoff_frr.png
- fig_tradeoff_utility.png
- fig_utility.png
