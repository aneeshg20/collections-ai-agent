# writing code by myself 

def risk_classifier (Invoice_number, days_overdue, dispute_flag, customer_type):
    if days_overdue > 60:
        return "HIGH"
    elif days_overdue > 35 or customer_type == "Low risk" or dispute_flag:
        return "MEDIUM"
    else:
        return "LOW"
    
# Testing it

print (f"Mckinsey invoice is {risk_classifier ("McKinsey",50,False,"High Risk")}")
print (f"Bain invoice is {risk_classifier ("Bain",40,False,"Low Risk")}")
print (f"BCG invoice is {risk_classifier ("BCG",25,False,"Medium Risk")}")