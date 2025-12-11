# 🔍 Fraud Detection Engine – Python (Rule-Based)

Motor de detección de fraude bancario basado en reglas, usando datos sintéticos y análisis transaccional.  
El objetivo es simular un sistema simple pero realista de monitoreo de operaciones para bancos y fintech.

---

## 🧠 ¿Qué hace este proyecto?

Este motor:

- Genera transacciones bancarias sintéticas (clientes, países, canales, saldos, MCC, IPs).
- Aplica reglas avanzadas de riesgo:
  - Montos extremadamente altos  
  - Países de alto riesgo (NG, RU, CN)  
  - IP diferente al país de origen de la tarjeta  
  - Actividad nocturna inusual (0–5 AM)  
  - Alta frecuencia de transacciones en ventana corta  
  - Vaciado casi total de la cuenta  
- Calcula un **risk score (0–100)** por transacción.
- Clasifica cada operación en **NONE / LOW / MEDIUM / HIGH risk**.
- Genera archivos CSV con resultados detallados.

---

## 📁 Archivos generados

- **transactions.csv** → todas las transacciones sin procesar  
- **transactions_scored.csv** → transacciones con riesgo, razones y etiquetas  
- **suspicious_transactions.csv** → solo las que fueron marcadas como sospechosas  

---

## ⚙️ Tecnologías utilizadas

- Python 3.x  
- pandas  
- numpy (opcional)  
- matplotlib (opcional)

---

## ▶️ Cómo ejecutar

Instalar dependencias:

```bash
pip install -r requirements.txt
