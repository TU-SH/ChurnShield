"""
generate_data.py
Generates synthetic Australian telco customer data and saves to:
  - data/raw/customers.csv   (for notebooks / training)
  - database/seed.sql        (for PostgreSQL seeding)

Run: python generate_data.py
"""
import random
import math
import csv
import os
from faker import Faker

fake = Faker('en_AU')
random.seed(42)

AU_STATES     = ['NSW','VIC','QLD','WA','SA','TAS','ACT','NT']
STATE_WEIGHTS  = [0.32, 0.26, 0.20, 0.10, 0.07, 0.02, 0.02, 0.01]
AREA_CODES     = {'NSW':['02'],'VIC':['03'],'QLD':['07'],
                  'WA':['08'],'SA':['08'],'TAS':['03'],'ACT':['02'],'NT':['08']}

N_CUSTOMERS = 3333

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def generate_customer(idx: int) -> dict:
    state = random.choices(AU_STATES, weights=STATE_WEIGHTS)[0]
    intl_plan = random.random() < 0.10
    vm_plan   = random.random() < 0.28
    acct_days = random.randint(1, 2500)
    cs_calls  = random.choices([0,1,2,3,4,5,6,7,8,9], weights=[30,25,18,10,6,4,3,2,1,1])[0]

    day_mins   = max(0, random.gauss(179.8, 54.5))
    day_calls  = max(0, int(random.gauss(100, 20)))
    day_charge = round(day_mins * 0.17, 2)

    eve_mins   = max(0, random.gauss(200.6, 50.1))
    eve_calls  = max(0, int(random.gauss(100, 20)))
    eve_charge = round(eve_mins * 0.085, 2)

    night_mins   = max(0, random.gauss(200.9, 54.2))
    night_calls  = max(0, int(random.gauss(100, 20)))
    night_charge = round(night_mins * 0.045, 2)

    intl_mins   = max(0, random.gauss(10.2, 7.9)) if intl_plan else max(0, random.gauss(2.1, 3.5))
    intl_calls  = max(0, int(random.gauss(4.5, 3.3))) if intl_plan else max(0, int(random.gauss(1.2, 2.0)))
    intl_charge = round(intl_mins * 0.27, 2)

    vm_msgs = max(0, int(random.gauss(8, 4))) if vm_plan else 0

    # Churn probability — driven by key signals known in AU telco
    total_charge = day_charge + eve_charge + night_charge + intl_charge
    churn_log = (
        -4.5
        + (cs_calls * 0.55)
        + (1.5 if intl_plan and intl_mins < 5 else 0)
        + (0.8 if total_charge > 75 else 0)
        + (-0.3 if vm_plan else 0)
        + (-0.002 * acct_days)
        + random.gauss(0, 0.4)
    )
    churn_prob = sigmoid(churn_log)
    churned = random.random() < churn_prob

    return {
        "customer_id":            f"AU-{idx:06d}",
        "state":                  state,
        "account_length_days":    acct_days,
        "area_code":              AREA_CODES[state][0],
        "international_plan":     intl_plan,
        "voicemail_plan":         vm_plan,
        "voicemail_messages":     vm_msgs,
        "day_mins":               round(day_mins, 2),
        "day_calls":              day_calls,
        "day_charge_aud":         day_charge,
        "evening_mins":           round(eve_mins, 2),
        "evening_calls":          eve_calls,
        "evening_charge_aud":     eve_charge,
        "night_mins":             round(night_mins, 2),
        "night_calls":            night_calls,
        "night_charge_aud":       night_charge,
        "intl_mins":              round(intl_mins, 2),
        "intl_calls":             intl_calls,
        "intl_charge_aud":        intl_charge,
        "customer_service_calls": cs_calls,
        "churned":                churned,
    }

def main():
    customers = [generate_customer(i+1) for i in range(N_CUSTOMERS)]
    churn_rate = sum(c["churned"] for c in customers) / len(customers)
    print(f"Generated {N_CUSTOMERS} customers | Churn rate: {churn_rate:.1%}")

    # CSV for training
    os.makedirs("data/raw", exist_ok=True)
    fields = list(customers[0].keys())
    with open("data/raw/customers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(customers)
    print("Saved: data/raw/customers.csv")

    # SQL seed file
    lines = ["-- Auto-generated seed data\n"]
    lines.append("INSERT INTO raw.customers (customer_id,state,account_length_days,area_code,")
    lines.append("  international_plan,voicemail_plan,voicemail_messages,")
    lines.append("  day_mins,day_calls,day_charge_aud,evening_mins,evening_calls,evening_charge_aud,")
    lines.append("  night_mins,night_calls,night_charge_aud,intl_mins,intl_calls,intl_charge_aud,")
    lines.append("  customer_service_calls,churned) VALUES")

    rows = []
    for c in customers:
        rows.append(
            f"  ('{c['customer_id']}','{c['state']}',{c['account_length_days']},'{c['area_code']}',"
            f"{str(c['international_plan']).upper()},{str(c['voicemail_plan']).upper()},{c['voicemail_messages']},"
            f"{c['day_mins']},{c['day_calls']},{c['day_charge_aud']},"
            f"{c['evening_mins']},{c['evening_calls']},{c['evening_charge_aud']},"
            f"{c['night_mins']},{c['night_calls']},{c['night_charge_aud']},"
            f"{c['intl_mins']},{c['intl_calls']},{c['intl_charge_aud']},"
            f"{c['customer_service_calls']},{str(c['churned']).upper()})"
        )

    lines.append(",\n".join(rows) + "\nON CONFLICT (customer_id) DO NOTHING;\n")

    with open("database/seed.sql", "w") as f:
        f.write("\n".join(lines))
    print("Saved: database/seed.sql")

if __name__ == "__main__":
    main()
