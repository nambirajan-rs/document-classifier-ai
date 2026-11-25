"""Generate synthetic invoices and bank statements for training.

Every name, company, address and number here is randomly generated, so the
training data (and the trained model) contains no real personal information.

Usage:
    python generate_data.py            # 150 of each type into data/
    python generate_data.py --n 300
"""
import argparse
import os
import random
from datetime import date, timedelta

FIRST = ["Arun", "Priya", "Karthik", "Divya", "Rahul", "Meena", "Vijay", "Anita",
         "Suresh", "Lakshmi", "John", "Sarah", "David", "Emma", "Ravi", "Nisha"]
LAST = ["Kumar", "Sharma", "Iyer", "Reddy", "Nair", "Smith", "Brown", "Patel",
        "Das", "Menon", "Wilson", "Rao"]
COMPANIES = ["Bluefin Traders", "Orbit Supplies", "Greenleaf Foods", "Nova Electronics",
             "Sunrise Textiles", "Apex Logistics", "Lotus Stationery", "Zenith Hardware",
             "Coral Software", "Maple Office Solutions", "Delta Packaging", "Vertex Pharma"]
BANKS = ["Example Bank", "Sample Savings Bank", "Demo National Bank", "Test Co-operative Bank",
         "Fictional Finance Bank", "Mock City Bank"]
CITIES = ["Chennai", "Trichy", "Madurai", "Coimbatore", "Bengaluru", "Hyderabad",
          "Pune", "Mumbai", "Kochi", "Salem"]
STREETS = ["MG Road", "Anna Salai", "Park Street", "Lake View Road", "Gandhi Nagar",
           "Station Road", "Temple Street", "Market Lane"]
ITEMS = ["A4 paper ream", "Laser printer toner", "Office chair", "USB keyboard",
         "Wireless mouse", "LED monitor 24 inch", "Packing boxes", "Hand sanitizer",
         "Software licence (annual)", "Network cable", "Desk lamp", "Whiteboard markers",
         "Consulting services", "Website maintenance", "Delivery charges", "Cloud hosting"]
DEBITS = ["ATM withdrawal", "UPI payment to grocery store", "Electricity bill payment",
          "Mobile recharge", "Online shopping", "Fuel station", "Restaurant",
          "Loan EMI", "Insurance premium", "Water bill", "DTH recharge", "Pharmacy"]
CREDITS = ["Salary credit", "Interest credited", "UPI received", "Cash deposit",
           "Refund", "NEFT transfer received", "Cheque deposit"]
TERMS = ["Payment due within 30 days.", "Net 15.", "Due on receipt.",
         "Please pay within 45 days of invoice date."]


def person():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def address():
    return f"{random.randint(1, 250)}, {random.choice(STREETS)}, {random.choice(CITIES)}"


def rand_date(start=date(2023, 1, 1), days=900):
    return start + timedelta(days=random.randint(0, days))


def fmt_date(d):
    return d.strftime(random.choice(["%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%Y-%m-%d"]))


def invoice():
    seller, buyer = random.sample(COMPANIES, 2)
    d = rand_date()
    lines, subtotal = [], 0.0
    for item in random.sample(ITEMS, random.randint(2, 7)):
        qty = random.randint(1, 20)
        price = round(random.uniform(50, 5000), 2)
        amount = qty * price
        subtotal += amount
        lines.append(f"{item:<28} {qty:>4} {price:>10.2f} {amount:>12.2f}")
    tax_rate = random.choice([5, 12, 18])
    tax = subtotal * tax_rate / 100
    title = random.choice(["TAX INVOICE", "INVOICE", "Invoice", "BILL / INVOICE"])
    return "\n".join([
        title,
        seller,
        address(),
        f"GSTIN: {random.randint(10, 37)}XXXXX{random.randint(1000, 9999)}X1Z{random.randint(1, 9)}",
        "",
        f"Invoice No: INV-{random.randint(1000, 99999)}",
        f"Invoice Date: {fmt_date(d)}",
        f"Due Date: {fmt_date(d + timedelta(days=random.choice([15, 30, 45])))}",
        "",
        "Bill To:",
        buyer,
        address(),
        "",
        f"{'Description':<28} {'Qty':>4} {'Unit Price':>10} {'Amount':>12}",
        *lines,
        "",
        f"Subtotal: {subtotal:.2f}",
        f"GST @ {tax_rate}%: {tax:.2f}",
        f"Total Amount Due: {subtotal + tax:.2f}",
        "",
        random.choice(TERMS),
        random.choice(["Thank you for your business.", "Authorised signatory",
                       "Bank transfer details provided on request."]),
    ])


def bank_statement():
    bank = random.choice(BANKS)
    start = rand_date()
    end = start + timedelta(days=random.choice([30, 31, 90]))
    balance = round(random.uniform(1000, 200000), 2)
    opening = balance
    rows = []
    d = start
    for _ in range(random.randint(6, 20)):
        d += timedelta(days=random.randint(0, 5))
        if random.random() < 0.35:
            desc, amt = random.choice(CREDITS), round(random.uniform(500, 60000), 2)
            balance += amt
            rows.append(f"{fmt_date(d):<12} {desc:<32} {'':>10} {amt:>10.2f} {balance:>12.2f}")
        else:
            desc, amt = random.choice(DEBITS), round(random.uniform(50, 15000), 2)
            balance -= amt
            rows.append(f"{fmt_date(d):<12} {desc:<32} {amt:>10.2f} {'':>10} {balance:>12.2f}")
    title = random.choice(["ACCOUNT STATEMENT", "Statement of Account", "BANK STATEMENT",
                           "Savings Account Statement"])
    return "\n".join([
        bank,
        title,
        "",
        f"Account Holder: {person()}",
        address(),
        f"Account Number: XXXXXX{random.randint(1000, 9999)}",
        f"Account Type: {random.choice(['Savings', 'Current', 'Salary'])}",
        f"Branch: {random.choice(CITIES)} Main",
        f"Statement Period: {fmt_date(start)} to {fmt_date(end)}",
        "",
        f"Opening Balance: {opening:.2f}",
        "",
        f"{'Date':<12} {'Description':<32} {'Debit':>10} {'Credit':>10} {'Balance':>12}",
        *rows,
        "",
        f"Closing Balance: {balance:.2f}",
        random.choice(["This is a computer generated statement.",
                       "Please report any discrepancy within 15 days.",
                       "Customer care is available 24x7."]),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=150, help="documents per class")
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        with open(os.path.join(args.out, f"invoice_{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(invoice())
        with open(os.path.join(args.out, f"bank_{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(bank_statement())
    print(f"Wrote {args.n * 2} synthetic documents to {args.out}/")


if __name__ == "__main__":
    main()
