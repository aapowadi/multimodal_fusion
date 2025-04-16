# Modus Operandi

1. Source data
    - Sentinel 1 [/work/mech-ai-scratch/rtali/gis-sentinel-1/final_s1]
    - Sentinel 2 [/work/mech-ai-scratch/rtali/gis-sentinel-2/final_s2_v3]

2. Output data
    - Sentinel 1 [imputation_time_based/output_images/imputed_s1]
    - Sentinel 2 [imputation_time_based/output_images/imputed_s2]

3. Run the following command to perform imputation:
```bash
python3 imputation_time_based/main.py s1
python3 imputation_time_based/main.py s2

```