# drought-predictor-neural-network

A drought predictor (Classification & Regression model) trained on climate features selectively extracted from the ERA5 global dataset over the period of 41 years from 1979 to 2020. Code is excluded for copyright reasons.

### Input Variables

| Variable | Description                                      |
|----------|--------------------------------------------------|
| mn2t     | Minimum temperature at 2 meters (°K)             |
| msl      | Mean sea level pressure (Pa)                     |
| mx2t     | Maximum temperature at 2 meters (°K)             |
| q        | Specific humidity (kg kg⁻¹)                      |
| t        | Average temperature at Pressure level 850 hPa (°K) |
| t2       | Average temperature at 2 meters (°K)             |
| tcc      | Total cloud cover (0-1)                          |
| u        | U wind component at pressure level 850 hPa (m s⁻¹) |
| u10      | U wind component at 10 meters (m s⁻¹)            |
| v        | V wind component at pressure level 850 hPa (m s⁻¹) |
| v10      | V wind component at 10 meters (m s⁻¹)            |
| z        | Geopotential (m² s⁻²)                            |
| month    | 1 to 12                                          |
| year     | 1979 to 2020                                     |
| grid ID  | The ID of the grid cell                          |
| SPI      | Standardised Precipitation Index (unitless)      |


### Target Variable

The drought index SPI (Standardised Precipitation Index) is used here as a proxy to characterise the intensity of drought caused by precipitation deficiency. Very low SPI values suggest intense drought, while very high values indicate very wet conditions. 
SPI is our target variable in the regression task, and we will predict it based on the climate variables. 
In the classification task, we will calculate a binary target variable ’Drought’ from SPI.

The ’Drought’ variable will be 1 to indicate the occurrence of drought and 0 to indicate no drought. We will apply a threshold of -1 to SPI, where values below or equal to this threshold indicate drought conditions (i.e. Drought = 1); otherwise, there is no drought (i.e. Drought = 0).

 time series plot of SPI for a single grid cell is shown in Fig. 3. Periods where SPI falls below or is equal to -1 indicate periods of drought.


### Preprocessing

 ```
Loading data
======= Climate_SPI.csv =======
Row, Col:  (15120, 16)
   year  month       u10       v10        mx2t        mn2t       tcc          t2            msl           t         q         u         v           z       SPI  grid_ID
0  1979      1 -2.958586 -0.634959  309.023720  297.283986  0.291988  302.992399  100796.767708  299.970613  0.008493 -4.568461 -2.839551  459.991748 -0.715037      303
1  1979     10 -0.049360 -0.113635  302.851483  289.459347  0.192154  296.133314  101346.796371  293.335081  0.004813 -2.372457 -3.721755  904.832294 -0.246504      303
2  1979     11 -0.404197 -0.341808  308.174168  294.707882  0.306703  301.354577  101135.556510  298.360025  0.005545 -2.794565 -3.715220  751.517519 -0.921090      303
3  1979     12 -1.089172 -0.600055  310.344489  296.948569  0.327175  303.544371  100999.285534  300.545961  0.006324 -3.429895 -3.687178  622.386996 -0.917537      303
4  1979      2 -2.160727 -0.715109  303.806692  294.093896  0.502479  298.759323  101125.883929  297.114917  0.009923 -3.892438 -2.943332  778.958111  0.631926      303
===============================
Excluding year and grid_ID from data
======= Climate_SPI.csv =======
Row, Col:  (15120, 14)
   month       u10       v10        mx2t        mn2t       tcc          t2            msl           t         q         u         v           z       SPI
0      1 -2.958586 -0.634959  309.023720  297.283986  0.291988  302.992399  100796.767708  299.970613  0.008493 -4.568461 -2.839551  459.991748 -0.715037
1     10 -0.049360 -0.113635  302.851483  289.459347  0.192154  296.133314  101346.796371  293.335081  0.004813 -2.372457 -3.721755  904.832294 -0.246504
2     11 -0.404197 -0.341808  308.174168  294.707882  0.306703  301.354577  101135.556510  298.360025  0.005545 -2.794565 -3.715220  751.517519 -0.921090
3     12 -1.089172 -0.600055  310.344489  296.948569  0.327175  303.544371  100999.285534  300.545961  0.006324 -3.429895 -3.687178  622.386996 -0.917537
4      2 -2.160727 -0.715109  303.806692  294.093896  0.502479  298.759323  101125.883929  297.114917  0.009923 -3.892438 -2.943332  778.958111  0.631926
===============================
Adding 'Drought' column (0 or 1) based on SPI
======= Climate_SPI.csv =======
Row, Col:  (2240, 15)
    month       u10       v10        mx2t        mn2t       tcc          t2            msl           t         q         u         v            z       SPI  Drought
9       7 -1.078143  0.902460  292.153382  278.805876  0.079997  285.057087  102283.450857  284.523974  0.003543 -2.473988 -1.114550  1695.127406 -1.099545        1
14     11 -0.753911 -0.576591  307.029682  293.860886  0.184344  300.264339  101172.854167  297.258291  0.005441 -3.071376 -3.726942   793.013513 -1.262581        1
20      6 -0.819211  0.867296  291.798410  280.497951  0.213060  285.672475  102054.188281  285.072712  0.005043 -2.170747 -0.765514  1537.134654 -1.606841        1
23      9 -0.170930  0.643260  301.150361  285.577702  0.054355  293.081444  101787.791667  291.516460  0.003437 -2.213144 -2.092395  1312.511619 -1.105385        1
37     10  0.034087  0.872630  302.460314  288.187859  0.148251  295.186908  101527.905746  292.523171  0.003713 -2.111258 -2.117658  1073.673213 -1.362772        1
===============================
Roughly 14.81% of the data has been labeled as 'Drought'.
The samples are equally distributed across 12 months over the entire dataset.
Month distribution: 
 month
1     1260
2     1260
3     1260
4     1260
5     1260
6     1260
7     1260
8     1260
9     1260
10    1260
11    1260
12    1260

Training set size: 10584 samples (70.0%)
Validation set size: 2268 samples (15.0%)
Test set size: 2268 samples (15.0%)
Total: 15120 samples

Replacing 'month' with 'cos_month' and 'sin_month'...
Training set: 
======= Climate_SPI.csv =======
Row, Col:  (10584, 16)
            u10       v10        mx2t        mn2t       tcc          t2            msl           t         q         u         v            z  Drought  month_normalised  cos(month_normalised)  sin(month_normalised)
6552  -1.479215 -0.349176  308.946537  296.729571  0.200873  302.857344  101032.544010  298.684343  0.007421 -3.278758 -2.823981   694.846833        0          0.000000           1.000000e+00               0.000000
10290  1.215222  1.341728  292.337389  282.794098  0.507675  287.565900  101984.340104  284.021999  0.005509 -0.961927 -0.816961  1463.148627        0          1.570796           6.123234e-17               1.000000
1098  -1.049909  0.459411  301.584464  289.574852  0.202775  295.267790  101702.975000  292.449862  0.006014 -2.335011 -1.833616  1250.147030        0          1.570796           6.123234e-17               1.000000
4525  -0.587676 -0.438277  301.028648  288.192378  0.286398  294.396633  101463.419607  290.270405  0.005417 -2.975901 -3.688355  1019.604987        0          4.712389          -1.836970e-16              -1.000000
4313  -1.335883  0.822051  304.262825  291.893402  0.207135  298.119280  101405.180696  293.967737  0.006583 -3.529105 -1.983720   983.308748        0          1.047198           5.000000e-01               0.866025
===============================

Validation set: 
======= Climate_SPI.csv =======
Row, Col:  (2268, 16)
            u10       v10        mx2t        mn2t       tcc          t2            msl           t         q         u         v            z  Drought  month_normalised  cos(month_normalised)  sin(month_normalised)
6706   1.839478 -0.081112  289.473332  278.865858  0.314341  283.883452  101895.747732  282.425177  0.003976 -0.153143 -1.877165  1323.057953        0          3.665191          -8.660254e-01              -0.500000
10093  0.897742  0.280632  295.250206  284.041292  0.427203  289.746081  101453.679940  284.966471  0.004620 -1.996432 -2.744730   945.190482        0          4.712389          -1.836970e-16              -1.000000
4169  -2.536848  0.726182  303.497044  293.499535  0.409161  298.526789  101526.751512  294.385840  0.007723 -4.536848 -2.243372  1103.272760        0          1.047198           5.000000e-01               0.866025
4660  -1.417957  0.659064  302.306398  289.817686  0.286100  296.016343  101274.662388  293.764555  0.007624 -2.855455 -1.091930   848.800585        0          0.523599           8.660254e-01               0.500000
5675   1.310550  0.547354  294.728937  282.397130  0.147330  288.758396  101700.329687  284.766466  0.004109 -1.696079 -2.510932  1189.587807        1          4.188790          -5.000000e-01              -0.866025
===============================
```

### Transformations

List of transformations applied:
- Removal of columns `year` and `grid_ID`
- Cyclic encoding of `month`, replacing it with `cos_month` and `sin_month`
- Standardisation of `u10`, `v10`, `mx2t`, `mn2t`, `tcc`, `t2`, `msl`, `t`, `q`, `u`, `v`, `z` using `StandardScaler`
- Skewness shifting using `PowerTransformer` (yeo-johanson) transformation on `q`, `tcc`, `v10`, `u`
  
![image](https://github.com/user-attachments/assets/fa72639d-d4b8-4070-a908-831192557290)

![image](https://github.com/user-attachments/assets/23035d36-8020-42d8-aa5b-9c370177aca6)


### Building the model

- A sequential model
- A single dense hidden layer (ReLU)
- Dropout = 0.3
- 100 epochs
- A single output layer (sigmoid)
- Loss function = Binary cross-entropy
- Batch sizes = [64, 128, 256]
- Optimisers = ['SGD', 'Adam']
- sgd_learning_rate = 0.01
- adam_learning_rate = 0.001

![image](https://github.com/user-attachments/assets/63b6e982-0c88-46d0-b4a6-e569cd83a61f)


- With SGD, the model with batch size 128 seems to be performing the best with lowest loss. 
- With Adam, the model with batch size 64 seems to be performing the best, and it achieves the lowest validation loss (~0.309) within 100 epochs. For both optimisers, the models with batch size 256 is converging more slowly and seems to require more than 100 epochs for it to yield results matching the other batch sizes.

The best model is with Adam with batch size 64, with the minimal validation loss 0.3048010468482971

![image](https://github.com/user-attachments/assets/841a3d7f-9f67-4133-b09d-7c558b02a90a)
![image](https://github.com/user-attachments/assets/fd15806f-ab63-45e3-9c5c-2d81016516cd)

The accuracy stabilizes around 86% for both training and validation sets. The validation accuracy initially appears slightly higher than training accuracy in the first 10-20 epochs, but falls afterwards to values slightly lower than training accuracy. The accuracy seems to stabilise around that point, though there is some fluctuation. I expect that there will be no significant increase in accuracy beyond this, indicating that the model has mostly converged at around 86%. 

### Optimisation

The 14 input features we have can be grouped by relevance. My hypothesis was that:
- `q` (humidity) and `tcc`(total cloud cover) are likely the most relevant factors to precipitation/droughtness,
- `cos(month_normalised)` and `sin(month_normalised)` might have a strong correlation due to seasonal nature of climate data,
- `t`, `t2`, `mx2t`, `mn2t` are all related to temperature, and using these all may be redundant, 
- `msl` (mean sea level), `u` (wind), `v` (wind), `u10` (wind), `v10` (wind) will have to be strongly correlated to each other, including temperature, in order to be an indicator of drought (i.e. for sea water to evaporate and form a rain cloud that can move quickly enough to reach the Murray-Darling basin area),

In addition to this grouping, we will train the model with these additional subsets:
- humidity + total cloud cover (`q`, `tcc`)
- humidity + seasonality (`q`, `cos(month_normalised)`, `sin(month_normalised)`)
- humidity + total cloud cover + seasonality (`q`, `tcc`, `cos(month_normalised)`, `sin(month_normalised)`)
- humidity + total cloud cover + seasonality + geopotential (`q`, `tcc`, `cos(month_normalised)`, `sin(month_normalised)`, `z`)
- temperature + msl + wind (`t`, `t2`, `mx2t`, `mn2t`, `msl`, `u`, `v`, `u10`, `v10`)
- temperature (average only) + msl + wind (`t`, `msl`, `u`, `v`, `u10`, `v10`)
- temperature + msl + wind + total cloud cover (`t`, `t2`, `mx2t`, `mn2t`, `msl`, `u`, `v`, `tcc`)
- temperature + msl + wind + total cloud cover + seasonality (`t`, `t2`, `mx2t`, `mn2t`, `msl`, `u`, `v`, `tcc`, `cos(month_normalised)`, `sin(month_normalised)`)
- temperature (average only) + msl + wind + humidity + total cloud cover + seasonality (`t`, `msl`, `u`, `v`, `q`, `tcc`, `cos(month_normalised)`, `sin(month_normalised)`)
- all 14 features

![image](https://github.com/user-attachments/assets/d5a31d23-6070-4488-9490-eaa25ed87829)

The best subset is: temp_avg_msl_wind_humidity_cloud_seasonality with validation accuracy: 0.86816579


### Evaluation

Accuracy: 0.81569665 (Balanced: 0.74676501)
Precision: 0.42084942
Recall: 0.64880952
F1-Score: 0.51053864

![image](https://github.com/user-attachments/assets/a2cb4e81-281b-4ab2-8340-e03d56c3d7de)


## Regression model

The goal is to predict the intensity of the drought, represented by ‘SPI’. We applied the same transformation and building configuation, except that the output layer uses linear activation and MSE for the loss function. (Metrics MAE)

![image](https://github.com/user-attachments/assets/16bb25b6-adc0-4783-b01c-feb267439438)

![image](https://github.com/user-attachments/assets/00b4aad4-4352-44c6-9e76-9c50cf8d36b0)

![image](https://github.com/user-attachments/assets/7e3588b3-b272-4738-aefd-21ba873f0256)

![image](https://github.com/user-attachments/assets/d7248e5b-9ce2-4901-8557-35c066592fc4)

![image](https://github.com/user-attachments/assets/5103f682-d95a-440e-bce1-6a22d1b0d569)

![image](https://github.com/user-attachments/assets/1450a59e-07fe-44af-ad9e-093f414e2b44)

Balanced Accuracy on New Data: 0.74614713
Precision on New Data: 0.44197685
Mean Absolute Error (MAE): 0.49075953
Pearson Correlation Coefficient: 0.78394778



 
