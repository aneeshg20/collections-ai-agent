from datetime import datetime
def payment_behaviour_classifier (vendor,invoice_receipt_date: datetime, invoice_due_date: datetime,invoice_payment_date: datetime,amount):
    if datetime.strptime(invoice_due_date,"%d-%b-%Y") < datetime.strptime(invoice_receipt_date,"%d-%b-%Y"):
        return "DEAD ON ARRIVAL"
    elif datetime.strptime(invoice_payment_date, "%d-%b-%Y") > datetime.strptime(invoice_due_date, "%d-%b-%Y"):
        return "Late Payment"
    elif datetime.strptime(invoice_payment_date, "%d-%b-%Y") < datetime.strptime(invoice_due_date, "%d-%b-%Y"):
        days_early = (datetime.strptime(invoice_due_date, "%d-%b-%Y") - datetime.strptime(invoice_payment_date, "%d-%b-%Y")).days
        wc_impact = (days_early * amount * 0.06) / 365
        print(f"Early Payment — Days early: {days_early} | WC Impact: ₹{wc_impact:,.0f}")
        return "Early Payment"
    else:
        return "ON TIME PAYMENT"
    

print(f"Siemens invoice is {payment_behaviour_classifier('Siemens','2-Mar-2026','2-Jun-2026','30-Apr-2026',500000)}")
print(f"Honeywell invoice is {payment_behaviour_classifier('Honeywell','1-Feb-2026','18-Mar-2026','20-Mar-2026',1200000)}")
print(f"ABB invoice is {payment_behaviour_classifier('ABB','1-Jan-2026','1-Mar-2026','15-Apr-2026',750000)}")