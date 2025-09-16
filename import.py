import pandas as pd
import mysql.connector
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# ----------- Step 1: Read CSV directly ------------
csv_file = "analytics_shipment.csv"

df = pd.read_csv(csv_file)
# Normalize column names: lowercase, strip, replace spaces/symbols with underscores
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r'[^\w\s]', '', regex=True)   # Remove parentheses, etc.
    .str.replace(r'\s+', '_', regex=True)      # Replace spaces with underscores
)

# Show cleaned column names
print("🧠 Normalized Columns:", df.columns.tolist())
print("📊 First few rows:")
print(df.head())
print("📋 Data types:")
print(df.dtypes)

# Show the actual CSV column names before normalization
print("🔍 Original CSV columns:")
with open(csv_file, 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    print("Raw header:", first_line)

# Clean and enrich DataFrame
df['created_at'] = pd.Timestamp.now()
df['updated_at'] = pd.Timestamp.now()

# Handle missing values and data type conversions
df['skus'] = df['skus'].fillna('').astype(str).replace(r'^\s*$', 'UNIDENTIFIED', regex=True)
df['therapy'] = df['therapy'].fillna('Unknown').astype(str)
df['record_type'] = df['type'].fillna('Unknown').astype(str)

# Clean numeric columns (remove commas)
df['quantity'] = df['quantity'].astype(str).str.replace(',', '').str.replace(' ', '')
df['valueinr'] = df['valueinr'].astype(str).str.replace(',', '').str.replace(' ', '')
df['wa_rate'] = df['wa_rate'].astype(str).str.replace(',', '').str.replace(' ', '')

# Convert to numeric
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype('int64')
df['value_inr'] = pd.to_numeric(df['valueinr'], errors='coerce').fillna(0.0)
df['wa_rate'] = pd.to_numeric(df['wa_rate'], errors='coerce').fillna(0.0)

# Convert date format
df['date'] = pd.to_datetime(df['date'], format='%d %b %y', errors='coerce')

# Fill all NaN values with appropriate defaults
df = df.fillna({
    'product_name': 'Unknown',
    'therapy': 'Unknown',
    'skus': 'UNIDENTIFIED',
    'indian_company': 'Unknown',
    'unit': 'Unknown',
    'city': 'Unknown',
    'foreign_company': 'Unknown',
    'foreign_country': 'Unknown',
    'continent': 'Unknown',
    'record_type': 'Unknown'
})

print("✅ CSV data loaded, cleaned, and calculations done.")

# ----------- Step 2: Import into MySQL ---------------
# Parse DB_URL for mysql.connector
db_url = os.getenv("DB_URL", "mysql+pymysql://root:@localhost:3306/venus_ai")
# Extract connection parameters from DB_URL
# Format: mysql+pymysql://user:password@host:port/database?charset=utf8mb4
import re
match = re.match(r'mysql\+pymysql://([^:]+):([^@]*)@([^:]+):(\d+)/([^?]+)', db_url)
if match:
    user, password, host, port, database = match.groups()
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=int(port)
    )
else:
    # Fallback to default values
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="venus_ai"
    )
cursor = conn.cursor()

# Create table with your new structure
cursor.execute("""
CREATE TABLE IF NOT EXISTS export_data (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  product_name    VARCHAR(255)       NULL,
  therapy         VARCHAR(120)       NULL,
  skus            VARCHAR(255)       NULL,
  date            DATE               NULL,
  indian_company  VARCHAR(255)       NULL,
  unit            VARCHAR(64)        NULL,
  city            VARCHAR(255)       NULL,
  foreign_company VARCHAR(255)       NULL,
  foreign_country VARCHAR(255)       NULL,
  continent       VARCHAR(64)        NULL,
  quantity        BIGINT             NULL,
  wa_rate         DECIMAL(18,3)      NULL,
  value_inr       DECIMAL(18,2)      NULL,
  record_type     VARCHAR(120)       NULL,
  created_at      TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_date (date),
  KEY idx_country (foreign_country),
  KEY idx_continent (continent),
  KEY idx_company (indian_company),
  KEY idx_foreign_company (foreign_company),
  KEY idx_product_date (product_name, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""")

# Import data using the cleaned DataFrame
print("🔍 Using DataFrame columns:", df.columns.tolist())

for i, row in df.iterrows():
    if i < 3:  # Debug first 3 rows
        print(f"🔍 Row {i+1} data:")
        for key, value in row.items():
            print(f"  {key}: {value}")
        print("---")
        
        # Debug what we're trying to insert
        print(f"🔍 Row {i+1} INSERT values:")
        print(f"  product_name: {row.get('product_name', None)}")
        print(f"  therapy: {row.get('therapy', None)}")
        print(f"  skus: {row.get('skus', None)}")
        print(f"  date: {row.get('date', None)}")
        print(f"  indian_company: {row.get('indian_company', None)}")
        print(f"  quantity: {row.get('quantity', None)}")
        print(f"  value_inr: {row.get('value_inr', None)}")
        print("---")
    
    cursor.execute("""
        INSERT INTO export_data (
            product_name, therapy, skus, date, indian_company, unit, city,
            foreign_company, foreign_country, continent, quantity, wa_rate,
            value_inr, record_type, created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """, (
        row.get('product_name', None),
        row.get('therapy', None),
        row.get('skus', None),
        row.get('date', None),
        row.get('indian_company', None),
        row.get('unit', None),
        row.get('city', None),
        row.get('foreign_company', None),
        row.get('foreign_country', None),
        row.get('continent', None),
        row.get('quantity', None),
        row.get('wa_rate', None),
        row.get('value_inr', None),
        row.get('record_type', None),
        row.get('created_at', None),
        row.get('updated_at', None)
    ))

conn.commit()
cursor.close()
conn.close()
print("✅ Data successfully imported to MySQL.")
