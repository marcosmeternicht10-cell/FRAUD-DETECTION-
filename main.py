import random
import csv
from datetime import datetime, timedelta

import pandas as pd


# -----------------------------
# 1. Generar datos sintéticos
# -----------------------------

def generate_synthetic_transactions(n_rows: int = 1000):
    random.seed(42)

    base_time = datetime.now() - timedelta(days=7)  # transacciones de la última semana

    customer_ids = list(range(1000, 1051))  # 51 clientes
    home_countries_map = {cid: random.choice(["PY", "AR", "BR"]) for cid in customer_ids}

    countries = ["PY", "AR", "BR", "US", "MX", "NG", "RU", "CN", "ES", "DE"]
    channels = ["APP", "WEB", "CAJERO", "POS", "SUCURSAL"]
    merchant_categories = [
        "SUPERMERCADO",
        "ELECTRONICA",
        "RESTAURANTE",
        "SUSCRIPCION_STREAMING",
        "CRIPTO_EXCHANGE",
        "CASINO_ONLINE",
        "LUJOS",
        "TRANSFERENCIA_P2P",
    ]

    rows = []
    for i in range(n_rows):
        tx_time = base_time + timedelta(minutes=random.randint(0, 60 * 24 * 7))
        customer = random.choice(customer_ids)
        country = random.choice(countries)
        channel = random.choice(channels)
        mcc = random.choice(merchant_categories)

        # montos base por categoría
        base_amount = {
            "SUPERMERCADO": random.uniform(20, 150),
            "ELECTRONICA": random.uniform(100, 1500),
            "RESTAURANTE": random.uniform(10, 200),
            "SUSCRIPCION_STREAMING": random.uniform(5, 25),
            "CRIPTO_EXCHANGE": random.uniform(200, 5000),
            "CASINO_ONLINE": random.uniform(50, 3000),
            "LUJOS": random.uniform(500, 10000),
            "TRANSFERENCIA_P2P": random.uniform(20, 4000),
        }[mcc]

        # algunos outliers bien locos
        if random.random() < 0.05:
            amount = round(base_amount * random.uniform(5, 15), 2)
        else:
            amount = round(base_amount * random.uniform(0.5, 1.5), 2)

        # saldo previo y nuevo
        previous_balance = round(random.uniform(0, 20000), 2)
        new_balance = max(previous_balance - amount, 0)

        # país de IP simulado (a veces distinto al país de la tarjeta)
        ip_country = random.choice(countries)
        home_country = home_countries_map[customer]

        row = {
            "tx_id": i + 1,
            "customer_id": customer,
            "home_country": home_country,
            "amount": amount,
            "country": country,  # país de la transacción
            "ip_country": ip_country,
            "channel": channel,
            "merchant_category": mcc,
            "previous_balance": previous_balance,
            "new_balance": new_balance,
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        rows.append(row)

    return rows


def save_transactions_to_csv(rows, filename="transactions.csv"):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# 2. Reglas avanzadas de fraude
# -----------------------------

def apply_fraud_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas de fraude y calcula un score de riesgo.
    """

    df = df.copy()
    df["is_suspicious"] = False
    df["reason"] = ""
    df["risk_score"] = 0  # 0 a 100

    # Convertimos timestamp a datetime
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp_dt"].dt.hour

    # Regla 1: monto muy alto
    high_amount_mask = df["amount"] > 8000
    df.loc[high_amount_mask, "is_suspicious"] = True
    df.loc[high_amount_mask, "reason"] += "Monto extremadamente alto; "
    df.loc[high_amount_mask, "risk_score"] += 40

    # Regla 2: país de riesgo
    risky_countries = ["NG", "RU", "CN"]
    risky_country_mask = df["country"].isin(risky_countries)
    df.loc[risky_country_mask, "is_suspicious"] = True
    df.loc[risky_country_mask, "reason"] += "País de alto riesgo; "
    df.loc[risky_country_mask, "risk_score"] += 25

    # Regla 3: IP en país distinto al país de tarjeta + transacción internacional
    mismatch_country_mask = (df["ip_country"] != df["home_country"]) & (df["country"] != df["home_country"])
    df.loc[mismatch_country_mask, "is_suspicious"] = True
    df.loc[mismatch_country_mask, "reason"] += "IP y tarjeta en países distintos; "
    df.loc[mismatch_country_mask, "risk_score"] += 20

    # Regla 4: transacciones nocturnas (0 a 5 AM) con monto medio/alto
    night_mask = df["hour"].between(0, 5) & (df["amount"] > 500)
    df.loc[night_mask, "is_suspicious"] = True
    df.loc[night_mask, "reason"] += "Actividad nocturna inusual; "
    df.loc[night_mask, "risk_score"] += 15

    # Regla 5: muchos intentos en ventana corta por cliente
    df = df.sort_values(by=["customer_id", "timestamp_dt"])

    window_minutes = 10
    max_tx_in_window = 6
    suspicious_idx = []

    for customer_id, group in df.groupby("customer_id"):
        times = group["timestamp_dt"].tolist()
        indices = group.index.tolist()

        start = 0
        for end in range(len(times)):
            while (times[end] - times[start]).total_seconds() / 60 > window_minutes:
                start += 1

            window_size = end - start + 1
            if window_size >= max_tx_in_window:
                suspicious_idx.extend(indices[start:end + 1])

    df.loc[suspicious_idx, "is_suspicious"] = True
    df.loc[suspicious_idx, "reason"] += f"Alta frecuencia en {window_minutes}min; "
    df.loc[suspicious_idx, "risk_score"] += 30

    # Regla 6: saldo casi en cero luego de un debito fuerte
    almost_zero_mask = (df["previous_balance"] > 0) & (df["new_balance"] < df["previous_balance"] * 0.05) & (df["amount"] > 1000)
    df.loc[almost_zero_mask, "is_suspicious"] = True
    df.loc[almost_zero_mask, "reason"] += "Vaciado de cuenta; "
    df.loc[almost_zero_mask, "risk_score"] += 20

    # Normalizar risk_score máximo 100
    df["risk_score"] = df["risk_score"].clip(0, 100)

    # Etiqueta de riesgo
    def label_risk(score):
        if score >= 70:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        elif score > 0:
            return "LOW"
        else:
            return "NONE"

    df["risk_label"] = df["risk_score"].apply(label_risk)

    return df


# -----------------------------
# 3. Reporting
# -----------------------------

def print_summary(df_result: pd.DataFrame):
    total = len(df_result)
    suspicious = df_result[df_result["is_suspicious"] == True]
    n_susp = len(suspicious)

    print("========== RESUMEN DE FRAUDE ==========")
    print(f"Transacciones totales: {total}")
    print(f"Transacciones sospechosas: {n_susp} ({(n_susp / total) * 100:.2f}%)")
    print()

    print("Por nivel de riesgo:")
    print(df_result["risk_label"].value_counts())
    print()

    print("Top 10 transacciones más riesgosas:")
    cols = [
        "tx_id",
        "customer_id",
        "amount",
        "country",
        "ip_country",
        "channel",
        "merchant_category",
        "risk_score",
        "risk_label",
        "reason",
    ]
    print(suspicious.sort_values(by="risk_score", ascending=False)[cols].head(10).to_string(index=False))


# -----------------------------
# 4. Main
# -----------------------------

def main():
    print("Generando transacciones sintéticas avanzadas...")
    rows = generate_synthetic_transactions(n_rows=1500)
    save_transactions_to_csv(rows, "transactions.csv")
    print("Archivo 'transactions.csv' generado.")

    print("Cargando datos y aplicando reglas de fraude avanzadas...")
    df = pd.read_csv("transactions.csv")

    df_result = apply_fraud_rules(df)

    # Guardar dataset completo con score
    df_result.to_csv("transactions_scored.csv", index=False)

    # Guardar solo sospechosas
    suspicious = df_result[df_result["is_suspicious"] == True]
    suspicious.to_csv("suspicious_transactions.csv", index=False)

    print("Archivos generados:")
    print("- transactions_scored.csv (todas las transacciones con score)")
    print("- suspicious_transactions.csv (solo las sospechosas)")
    print()

    print_summary(df_result)


if __name__ == "__main__":
    main()
