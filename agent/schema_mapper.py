import pandas as pd
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

# The schema our pipeline requires - what every invoice MUST have
REQUIRED_SCHEMA = {
        "vendor": "Name of the vendor/customer/company the invoice is for",
        "invoice_amount": "The monetary amount of the invoice (number)",
        "days_since_invoice": "How many days since the invoice was raised (number)",
        "payment_term_days": "The agreed payment term in days (number)"
}

def detect_mapping(df):
    """Uses Claude to map the uploaded CSV's columns to required schema"""

    #Give Claude the column names + a few sample rows to reason over

    columns = list(df.columns)
    sample_rows = df.head(3).to_dict(orient = "records")

    system_prompt = f"""You are a Schema Mapping Agent. Your job is to map columns from an uploaded CSV to a required target schema.

TARGET SCHEMA (what we need):
{json.dumps(REQUIRED_SCHEMA, indent=2)}

UPLOADED CSV COLUMNS:
{columns}

SAMPLE DATA (first 3 rows):
{json.dumps(sample_rows, indent=2, default=str)}

YOUR TASK:
Map each target schema field to the most appropriate uploaded column.
- If a target field has a clear match, map it.
- If a target field has NO reasonable match in the uploaded data, set it to null.
- Do NOT guess or force a mapping if the data genuinely isn't there.

Respond with ONLY a JSON object, no other text, in this exact format:
{{
  "vendor": "uploaded_column_name_or_null",
  "invoice_amount": "uploaded_column_name_or_null",
  "days_since_invoice": "uploaded_column_name_or_null",
  "payment_term_days": "uploaded_column_name_or_null"
}}"""
    
    message = client.messages.create(
        model = "claude-sonnet-4-5",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": "Map the columns and return the JSON."}]
    )

    response_text = message.content[0].text.strip()

    #Strip markdown fences if Claude wrapped the JSON in them
    response_text = response_text.replace("```json", "").replace("```", "").strip()

    mapping = json.loads(response_text)
    return mapping

def validate_mapping(mapping):
    """Checks if all required fields were successfully mapped.
    Returns (is_sufficient, missing_fields)."""
    
    missing = []
    for field, source_column in mapping.items():
        # Claude returns null (None in Python) for unmappable fields
        if source_column is None or source_column == "null":
            missing.append(field)
    
    is_sufficient = len(missing) == 0
    return is_sufficient, missing


def apply_mapping(df, mapping):
    """Renames the uploaded columns to our schema names."""
    # Build a rename dict: {uploaded_column: our_field}
    rename_dict = {source: target for target, source in mapping.items() if source}
    mapped_df = df.rename(columns=rename_dict)
    # Keep only the columns we need
    mapped_df = mapped_df[list(mapping.keys())]
    return mapped_df

# Test it
if __name__ == "__main__":
    df = pd.read_csv("agent/messy_invoices.csv")
    print("Uploaded columns:", list(df.columns))
    print("\nDetecting mapping via Claude...\n")
    
    mapping = detect_mapping(df)
    print("Detected mapping:")
    for target, source in mapping.items():
        print(f"  {target:<20} <- {source}")
    
    # Validate sufficiency
    is_sufficient, missing = validate_mapping(mapping)
    
    print("\n" + "=" * 50)
    if is_sufficient:
        print("✅ SUFFICIENT DATA - all required fields mapped")
        mapped_df = apply_mapping(df, mapping)
        print("\nMapped dataframe (ready for pipeline):")
        print(mapped_df.head())
    else:
        print(f"❌ INSUFFICIENT DATA - cannot proceed")
        print(f"   Missing required fields: {missing}")
        print(f"   The uploaded CSV does not contain data for these fields.")
        print(f"   Pipeline will NOT run to avoid processing incomplete data.")