# Crypto_Volatility_prediction
We propose two types pf novel hybrid models that combine the strengths of GARCH-type models with GRU neural networks to enhance the volatility forecasting in cryptocurrency markets.

📌 **Hybrid Cryptocurrency Volatility Prediction: GARCH + GRU**
This repository presents an advanced framework for cryptocurrency volatility forecasting, combining traditional GARCH-type models with Gated Recurrent Unit (GRU) networks. By leveraging both econometric modeling and deep learning, this approach enhances volatility prediction accuracy across short-term and long-term horizons.

---

🔹 **Project Overview**
Traditional volatility forecasting models, such as AVARCH and FIGARCH, often struggle to capture the nonlinear and high-frequency characteristics of cryptocurrency markets. This project introduces two hybrid correction frameworks:
1. **GRU-GARCH Model**: Utilizes GARCH volatility estimates and residuals as inputs to a GRU network for enhanced predictions.
2. **GRU-GARCH-Error Model**: Predicts and corrects GARCH model errors via GRU to refine volatility forecasts dynamically.

---

🔹 **Key Contributions**
✔️ Data preprocessing and segmentation using a sliding window approach  
✔️ Implementation of GARCH-type models (AVARCH, FIGARCH) for variance prediction  
✔️ Integration of GRU models for capturing sequential dependencies  
✔️ Error-based GRU-GARCH hybrid modeling for volatility correction  
✔️ Empirical evaluation of hybrid models against standalone GARCH and GRU models  

---

🚀 **Workflow**

<img width="232" alt="image" src="https://github.com/user-attachments/assets/394fc12f-4153-4dbe-8f24-6e80d98a3154" />     <img width="422" alt="image" src="https://github.com/user-attachments/assets/ee2a0f3c-7314-4394-8a74-731bbe2b0371" />

1️⃣ **Data Preprocessing & Segmentation**  
- Slect Top10 traded cryptos and form a portfolio
- Splits portfolio data into 20% validation, 60% training, and 20% testing sets  
- Uses a sliding window of 120 data points with one-step shifts, resulting in **M = 6861 windows**  
- Computes realized variance and trading volume as additional features  

2️⃣ **GARCH Model Training & Bootstrap Simulation**  
- Implements AVARCH, FIGARCH, and standard GARCH models  
- Predicts variance for h=1, then applies bootstrap simulation for h=2 to h=6  

3️⃣ **GRU Model Training**  
- Uses a supervised learning approach with variance data as inputs  
- Conducts hyperparameter tuning (Grid Search & Cross-validation)  
- Optimizes with **Huber loss function** and **Adam optimizer**  

4️⃣ **Hybrid Model Integration**  
- **GRU-GARCH Model**: Uses GARCH volatility, residuals, realized variance, and trading volume as GRU inputs  
- **GRU-GARCH-Error Model**: Trains GRU to predict GARCH model errors and refines variance predictions  
- Adapts rolling window training for dynamic market conditions  

---

📈 **Results & Performance**
✔️ GRU-GARCH hybrid models outperform traditional GARCH-type models in volatility forecasting  
✔️ The GRU-GARCH model (AVARCH variant) provides the highest accuracy in both short-term and long-term horizons  
✔️ Error-based GRU-GARCH hybrid models did not achieve comparable performance  
✔️ Analysis of convergence properties explains sensitivity differences across GARCH-type models  

---

🏆 **Key Features**
✔️ Full Pipeline: Data Cleaning → GARCH Training → GRU Correction  
✔️ High-Frequency Cryptocurrency Volatility Forecasting  
✔️ Rolling Window Adaptation for Market Changes  
✔️ Python-based Implementation with TensorFlow, Keras, NumPy & QuantLib  

---

📬 **Contact & Citation**
If you find this work helpful, feel free to reach out via **k3802286782@gmail.com \ tenghanz@usc.edu** or contribute to the repository!

