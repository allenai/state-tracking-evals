| term | cube_task | formula | glm_cov_type | glm_logit_coef | glm_logit_se | glm_p | wls_cov_type | wls_accuracy_coef | wls_accuracy_se | wls_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | unique | score ~ C(architecture) + num_variables_centered | HC3 | 0.941886 | 0.00307865 | 0 | HC3 | 0.710333 | 0.0568504 | 7.97056e-36 |
| C(architecture)[T.GDN± 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.00168994 | 0.00395293 | 0.669003 | HC3 | -0.000333333 | 0.0739472 | 0.996403 |
| C(architecture)[T.hybrid 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 0.204437 | 0.00427502 | 0 | HC3 | 0.0386667 | 0.0629066 | 0.538774 |
| C(architecture)[T.hybrid 275M] | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.219391 | 0.00365085 | 0 | HC3 | -0.045 | 0.0821889 | 0.584022 |
| C(architecture)[T.hybrid 7B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 4.11602 | 0.0108801 | 0 | HC3 | 0.282667 | 0.0829674 | 0.000656905 |
| C(architecture)[T.transformer 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 1.11159 | 0.00542286 | 0 | HC3 | 0.167667 | 0.0673736 | 0.0128243 |
| C(architecture)[T.transformer 7B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 2.86322 | 0.0262912 | 0 | HC3 | 0.265667 | 0.0991034 | 0.00734682 |
| num_variables_centered | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.283584 | 0.00103796 | 0 | HC3 | -0.0376429 | 0.0154065 | 0.0145532 |
| Intercept | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.180995 | 0.00918851 | 2.24376e-86 | HC3 | 0.543975 | 0.0103144 | 0 |
| C(architecture)[T.GDN± 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.401294 | 0.0131832 | 1.62888e-203 | HC3 | -0.0973333 | 0.014738 | 3.99503e-11 |
| C(architecture)[T.hybrid 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.144296 | 0.0125944 | 2.16401e-30 | HC3 | -0.035 | 0.0141154 | 0.0131547 |
| C(architecture)[T.hybrid 275M] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.295106 | 0.0124928 | 2.28418e-123 | HC3 | -0.0716667 | 0.0141878 | 4.3885e-07 |
| C(architecture)[T.hybrid 7B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0399682 | 0.0153889 | 0.00939861 | HC3 | -0.00966667 | 0.0177629 | 0.5863 |
| C(architecture)[T.transformer 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.0498052 | 0.0126859 | 8.63647e-05 | HC3 | 0.012 | 0.0141114 | 0.395116 |
| C(architecture)[T.transformer 7B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.394651 | 0.0142116 | 1.01099e-169 | HC3 | 0.0926667 | 0.0153735 | 1.66328e-09 |
| num_variables_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.11451 | 0.00224496 | 0 | HC3 | -0.0275449 | 0.002517 | 7.13446e-28 |
| num_assignments_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0189598 | 0.000283335 | 0 | HC3 | -0.00457086 | 0.000321162 | 5.77972e-46 |
| num_variables_centered:num_assignments_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.000176341 | 0.000171514 | 0.303881 | HC3 | -7.29996e-05 | 0.000194454 | 0.707357 |
| Intercept | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.115308 | 0.00855919 | 2.29e-41 | HC3 | 0.471774 | 0.00956459 | 0 |
| C(architecture)[T.GDN± 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.264283 | 0.0118694 | 7.91097e-110 | HC3 | -0.063 | 0.01316 | 1.69108e-06 |
| C(architecture)[T.hybrid 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0373126 | 0.0126708 | 0.00323199 | HC3 | -0.009 | 0.0141783 | 0.525578 |
| C(architecture)[T.hybrid 275M] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.115066 | 0.0126519 | 9.47483e-20 | HC3 | -0.0276667 | 0.0140154 | 0.0483803 |
| C(architecture)[T.hybrid 7B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.111543 | 0.0129153 | 5.79591e-18 | HC3 | 0.027 | 0.0148119 | 0.0683241 |
| C(architecture)[T.transformer 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0276298 | 0.0127109 | 0.0297268 | HC3 | -0.00666667 | 0.0143344 | 0.641874 |
| C(architecture)[T.transformer 7B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.311484 | 0.0117339 | 2.88655e-155 | HC3 | -0.074 | 0.0128855 | 9.30849e-09 |
| num_variables_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.113843 | 0.00213677 | 0 | HC3 | -0.0274513 | 0.00237239 | 5.77064e-31 |
| num_assignments_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0220769 | 0.000260352 | 0 | HC3 | -0.00532393 | 0.000284346 | 3.18849e-78 |
| num_variables_centered:num_assignments_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.00126121 | 0.000172591 | 2.72213e-13 | HC3 | 0.000355111 | 0.000186525 | 0.0569331 |
| Intercept | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.228884 | 0.00765429 | 1.81345e-196 | HC3 | 0.444587 | 0.00848034 | 0 |
| C(architecture)[T.GDN± 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.24356 | 0.0110436 | 8.65347e-108 | HC3 | -0.0576667 | 0.0120875 | 1.83514e-06 |
| C(architecture)[T.hybrid 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0568965 | 0.0110836 | 2.8454e-07 | HC3 | -0.0136667 | 0.0121716 | 0.26151 |
| C(architecture)[T.hybrid 275M] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.20342 | 0.0107651 | 1.22539e-79 | HC3 | -0.0483333 | 0.0118572 | 4.57612e-05 |
| C(architecture)[T.hybrid 7B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.178496 | 0.012137 | 5.82651e-49 | HC3 | 0.0433333 | 0.0136499 | 0.00150031 |
| C(architecture)[T.transformer 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.0331367 | 0.010993 | 0.00257542 | HC3 | 0.008 | 0.0123495 | 0.517115 |
| C(architecture)[T.transformer 7B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.245 | 0.011711 | 3.48235e-97 | HC3 | -0.058 | 0.0130161 | 8.34994e-06 |
| num_variables_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.133885 | 0.00194999 | 0 | HC3 | -0.031633 | 0.00215139 | 6.12061e-49 |
| num_assignments_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0142424 | 0.000223098 | 0 | HC3 | -0.00334243 | 0.000248463 | 2.97954e-41 |
| num_variables_centered:num_assignments_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.00381651 | 0.000141344 | 1.41646e-160 | HC3 | -0.000821956 | 0.000155276 | 1.19978e-07 |
