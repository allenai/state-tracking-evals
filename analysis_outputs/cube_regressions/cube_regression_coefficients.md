| term | cube_task | formula | glm_cov_type | glm_logit_coef | glm_logit_se | glm_p | wls_cov_type | wls_accuracy_coef | wls_accuracy_se | wls_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intercept | unique | score ~ C(architecture) + num_variables_centered | HC3 | 0.946069 | 0.0028623 | 0 | HC3 | 0.710333 | 0.0403549 | 2.37042e-69 |
| C(architecture)[T.GDN± 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.00169645 | 0.00368711 | 0.645443 | HC3 | -0.000333333 | 0.0510089 | 0.994786 |
| C(architecture)[T.hybrid 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 0.205187 | 0.00421564 | 0 | HC3 | 0.0386667 | 0.0460778 | 0.40138 |
| C(architecture)[T.transformer 1B] | unique | score ~ C(architecture) + num_variables_centered | HC3 | 1.11477 | 0.00558017 | 0 | HC3 | 0.167667 | 0.0738883 | 0.0232568 |
| num_variables_centered | unique | score ~ C(architecture) + num_variables_centered | HC3 | -0.296576 | 0.000984101 | 0 | HC3 | -0.0505625 | 0.0150117 | 0.000756605 |
| Intercept | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.181689 | 0.00875362 | 1.08559e-95 | HC3 | 0.54445 | 0.00991528 | 0 |
| C(architecture)[T.GDN± 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.397577 | 0.0124801 | 1.05542e-222 | HC3 | -0.0973333 | 0.0140687 | 4.56663e-12 |
| C(architecture)[T.hybrid 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.14295 | 0.0123437 | 5.15374e-31 | HC3 | -0.035 | 0.0139869 | 0.0123375 |
| C(architecture)[T.transformer 1B] | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.0493413 | 0.0121016 | 4.55702e-05 | HC3 | 0.012 | 0.0136244 | 0.378441 |
| num_variables_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.128147 | 0.00268684 | 0 | HC3 | -0.0314009 | 0.00301093 | 1.82832e-25 |
| num_assignments_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0109193 | 0.000309172 | 3.14105e-273 | HC3 | -0.00266459 | 0.000353574 | 4.84019e-14 |
| num_variables_centered:num_assignments_centered | reassign const | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.000838097 | 0.000192265 | 1.30619e-05 | HC3 | 0.000191817 | 0.000220426 | 0.384186 |
| Intercept | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.114764 | 0.00847027 | 8.03194e-42 | HC3 | 0.471837 | 0.00949689 | 0 |
| C(architecture)[T.GDN± 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.263739 | 0.0117757 | 4.21257e-111 | HC3 | -0.063 | 0.0130941 | 1.49934e-06 |
| C(architecture)[T.hybrid 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0372345 | 0.0127241 | 0.00343013 | HC3 | -0.009 | 0.0143114 | 0.529435 |
| C(architecture)[T.transformer 1B] | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0275719 | 0.0125231 | 0.0276878 | HC3 | -0.00666667 | 0.014191 | 0.638511 |
| num_variables_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.122921 | 0.00276356 | 0 | HC3 | -0.0297856 | 0.00307217 | 3.15731e-22 |
| num_assignments_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0201638 | 0.00033726 | 0 | HC3 | -0.00488223 | 0.000372164 | 2.57952e-39 |
| num_variables_centered:num_assignments_centered | reassign var | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.00146144 | 0.000220588 | 3.46818e-11 | HC3 | 0.000401106 | 0.000241111 | 0.0961976 |
| Intercept | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.228387 | 0.00764661 | 5.17158e-196 | HC3 | 0.444687 | 0.00851017 | 0 |
| C(architecture)[T.GDN± 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.243358 | 0.0110352 | 8.95792e-108 | HC3 | -0.0576667 | 0.0121259 | 1.97802e-06 |
| C(architecture)[T.hybrid 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.056849 | 0.0110747 | 2.8484e-07 | HC3 | -0.0136667 | 0.0122049 | 0.262813 |
| C(architecture)[T.transformer 1B] | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | 0.033109 | 0.0109713 | 0.00254643 | HC3 | 0.008 | 0.0123711 | 0.517847 |
| num_variables_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.13234 | 0.00250392 | 0 | HC3 | -0.031403 | 0.00277152 | 9.25421e-30 |
| num_assignments_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.0136373 | 0.000288517 | 0 | HC3 | -0.00321013 | 0.000320096 | 1.14074e-23 |
| num_variables_centered:num_assignments_centered | undefined | score ~ C(architecture) + num_variables_centered * num_assignments_centered | HC3 | -0.004084 | 0.000188144 | 1.77288e-104 | HC3 | -0.000894991 | 0.000207076 | 1.54585e-05 |
