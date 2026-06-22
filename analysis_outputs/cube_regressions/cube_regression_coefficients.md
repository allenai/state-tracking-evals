| term | cube_task | formula | glm_cov_type | glm_logit_coef | glm_logit_se | glm_p | wls_cov_type | wls_accuracy_coef | wls_accuracy_se | wls_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | unique | score ~ C(architecture) + num_variables_centered | HC3 | 0.942471 | 0.00301649 | 0 | HC3 | 0.710333 | 0.0517363 | 6.72312e-43 |
| C(architecture)[T.GDN± 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.00169085 | 0.00391171 | 0.665556 | HC3 | -0.000333333 | 0.0669948 | 0.99603 |
| C(architecture)[T.hybrid 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 0.204542 | 0.00426564 | 0 | HC3 | 0.0386667 | 0.057125 | 0.498483 |
| C(architecture)[T.hybrid 275M] | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.219515 | 0.00359255 | 0 | HC3 | -0.045 | 0.0748817 | 0.547875 |
| C(architecture)[T.hybrid 7B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 4.11676 | 0.010913 | 0 | HC3 | 0.282667 | 0.0822152 | 0.000585745 |
| C(architecture)[T.transformer 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 1.11203 | 0.00545545 | 0 | HC3 | 0.167667 | 0.0653469 | 0.010294 |
| C(architecture)[T.transformer 275M] | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.0635074 | 0.00302426 | 6.65625e-98 | HC3 | -0.0126667 | 0.0600952 | 0.833061 |
| C(architecture)[T.transformer 7B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 2.86391 | 0.0263104 | 0 | HC3 | 0.265667 | 0.0989253 | 0.00724154 |
| num_variables_centered | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.285436 | 0.000849 | 0 | HC3 | -0.0404375 | 0.0136297 | 0.00300842 |
| Intercept | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.180704 | 0.00913815 | 4.92046e-87 | HC3 | 0.543846 | 0.0102569 | 0 |
| C(architecture)[T.GDN± 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.401171 | 0.0131152 | 1.75521e-205 | HC3 | -0.0973333 | 0.0146527 | 3.07988e-11 |
| C(architecture)[T.hybrid 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.144245 | 0.0125595 | 1.57048e-30 | HC3 | -0.035 | 0.0140754 | 0.0128968 |
| C(architecture)[T.hybrid 275M] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.295011 | 0.0124147 | 8.07783e-125 | HC3 | -0.0716667 | 0.0140905 | 3.65374e-07 |
| C(architecture)[T.hybrid 7B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0399535 | 0.0154188 | 0.00956359 | HC3 | -0.00966667 | 0.017797 | 0.587018 |
| C(architecture)[T.transformer 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.0497862 | 0.0126343 | 8.12924e-05 | HC3 | 0.012 | 0.0140549 | 0.393218 |
| C(architecture)[T.transformer 275M] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.395633 | 0.0122203 | 6.18582e-230 | HC3 | -0.096 | 0.0137098 | 2.51801e-12 |
| C(architecture)[T.transformer 7B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.394483 | 0.0142257 | 3.0054e-169 | HC3 | 0.0926667 | 0.0153986 | 1.76742e-09 |
| num_variables_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.116371 | 0.00205678 | 0 | HC3 | -0.0280148 | 0.00230807 | 6.6619e-34 |
| num_assignments_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0185927 | 0.000258398 | 0 | HC3 | -0.00448439 | 0.000292678 | 5.45868e-53 |
| num_variables_centered:num_assignments_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.000238217 | 0.000156445 | 0.127835 | HC3 | 3.60555e-05 | 0.00017738 | 0.838926 |
| Intercept | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.115114 | 0.00851557 | 1.22277e-41 | HC3 | 0.471833 | 0.00951356 | 0 |
| C(architecture)[T.GDN± 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.264174 | 0.0118409 | 2.93441e-110 | HC3 | -0.063 | 0.0131246 | 1.58541e-06 |
| C(architecture)[T.hybrid 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0372975 | 0.0126828 | 0.00327375 | HC3 | -0.009 | 0.014194 | 0.526034 |
| C(architecture)[T.hybrid 275M] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.11502 | 0.0125891 | 6.45226e-20 | HC3 | -0.0276667 | 0.0139436 | 0.0472356 |
| C(architecture)[T.hybrid 7B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.111499 | 0.0129481 | 7.2284e-18 | HC3 | 0.027 | 0.0148513 | 0.0690608 |
| C(architecture)[T.transformer 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0276187 | 0.0126339 | 0.0288102 | HC3 | -0.00666667 | 0.0142474 | 0.639839 |
| C(architecture)[T.transformer 275M] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0899746 | 0.012376 | 3.59278e-13 | HC3 | -0.0216667 | 0.0137827 | 0.115947 |
| C(architecture)[T.transformer 7B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.311355 | 0.0117016 | 5.5091e-156 | HC3 | -0.074 | 0.0128469 | 8.40468e-09 |
| num_variables_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.11835 | 0.00200012 | 0 | HC3 | -0.0285505 | 0.00222111 | 8.14889e-38 |
| num_assignments_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0215621 | 0.000245621 | 0 | HC3 | -0.00520238 | 0.000268129 | 7.34537e-84 |
| num_variables_centered:num_assignments_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.00109929 | 0.000163399 | 1.72389e-11 | HC3 | 0.00031873 | 0.000176469 | 0.0708936 |
| Intercept | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.228774 | 0.00765526 | 3.12362e-196 | HC3 | 0.444643 | 0.00847542 | 0 |
| C(architecture)[T.GDN± 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.243623 | 0.0110483 | 9.37199e-108 | HC3 | -0.0576667 | 0.0120868 | 1.83286e-06 |
| C(architecture)[T.hybrid 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0569121 | 0.0110886 | 2.85955e-07 | HC3 | -0.0136667 | 0.0121686 | 0.261391 |
| C(architecture)[T.hybrid 275M] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.203473 | 0.0107682 | 1.23426e-79 | HC3 | -0.0483333 | 0.0118553 | 4.56303e-05 |
| C(architecture)[T.hybrid 7B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.178548 | 0.0121429 | 6.08467e-49 | HC3 | 0.0433333 | 0.0136473 | 0.00149716 |
| C(architecture)[T.transformer 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.033146 | 0.0109892 | 0.00255941 | HC3 | 0.008 | 0.0123368 | 0.516683 |
| C(architecture)[T.transformer 275M] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.111456 | 0.0110622 | 7.09742e-24 | HC3 | -0.0266667 | 0.0122368 | 0.0293151 |
| C(architecture)[T.transformer 7B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.245064 | 0.0117099 | 2.98205e-97 | HC3 | -0.058 | 0.0130088 | 8.25283e-06 |
| num_variables_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.134984 | 0.00182162 | 0 | HC3 | -0.0318686 | 0.00200928 | 1.18559e-56 |
| num_assignments_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0142404 | 0.00020805 | 0 | HC3 | -0.00333763 | 0.000231361 | 3.54612e-47 |
| num_variables_centered:num_assignments_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.00394858 | 0.000132166 | 4.03695e-196 | HC3 | -0.000850831 | 0.000145057 | 4.47774e-09 |
