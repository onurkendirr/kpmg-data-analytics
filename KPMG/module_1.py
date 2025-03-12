import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = "dataset/KPMG_VI_New_raw_data_update_final.xlsx"
print(pd.ExcelFile(data).sheet_names)

############Customer Demographic############

cust_demo_df = pd.read_excel(data, header=1, sheet_name="CustomerDemographic")
cust_demo_df.head()

cust_demo_df.shape

cust_demo_df[cust_demo_df.duplicated()].sum()
print("customer_id blanks:", pd.isna(cust_demo_df["customer_id"]).sum())

pd.notna(cust_demo_df["customer_id"].unique()).sum()

print("gender:", cust_demo_df["gender"].unique())
print("blanks:", pd.isna(cust_demo_df["gender"]).sum())
plt.hist(cust_demo_df["gender"][pd.notna(cust_demo_df["gender"])], bins=6, edgecolor="black", align="mid")
plt.show()

cust_demo_df["past_3_years_bike_related_purchases"].describe()
cust_demo_df["DOB"].describe()
print("DOB blanks:", pd.isna(cust_demo_df["DOB"]).sum())

cust_demo_df["DOB"] = pd.to_datetime(cust_demo_df["DOB"])
plt.scatter([d.year for d in cust_demo_df["DOB"]], [d.month for d in cust_demo_df["DOB"]])
plt.show()

from datetime import datetime
cust_demo_df["age"] = (datetime.now() - cust_demo_df["DOB"]) // 365

cust_demo_df["age"].describe()

plt.scatter([d.year for d in cust_demo_df["DOB"]], cust_demo_df["age"].dt.days)
plt.show()

print("job_title:", cust_demo_df["job_title"].unique())
print("blanks:", pd.isna(cust_demo_df["job_title"]).sum())

print("job_industry_category:", cust_demo_df["job_industry_category"].unique())
print("blanks:", pd.isna(cust_demo_df["job_industry_category"]).sum())

print("wealth_segment:", cust_demo_df["wealth_segment"].unique())
print("blanks:", pd.isna(cust_demo_df["wealth_segment"]).sum())
plt.hist(cust_demo_df["wealth_segment"][pd.notna(cust_demo_df["wealth_segment"])], bins=3, edgecolor="black", align="mid")
plt.show()

print('deceased_indicator:', cust_demo_df['deceased_indicator'].unique())
print('blanks:', pd.isna(cust_demo_df['deceased_indicator']).sum())
plt.hist(cust_demo_df['deceased_indicator'][pd.notna(cust_demo_df['deceased_indicator'])], bins=2, edgecolor="black", align="mid")
plt.show()

cust_demo_df['deceased_indicator'][cust_demo_df['deceased_indicator'] == 'Y'].count()

print('owns_car:', cust_demo_df['owns_car'].unique())
print('blanks:', pd.isna(cust_demo_df['owns_car']).sum())
plt.hist(cust_demo_df['owns_car'][pd.notna(cust_demo_df['owns_car'])], bins=2, edgecolor="black", align="mid")
plt.show()


cust_demo_df['tenure'].describe()

print('tenure blanks:', pd.isna(cust_demo_df['tenure']).sum())

####################Customer Adress#############################

cust_addr_df = pd.read_excel(data, header=1, sheet_name="CustomerAddress")
cust_addr_df.head()

cust_addr_df.shape

cust_addr_df[cust_addr_df.duplicated()].sum()
print("customer_id blanks:", pd.isna(cust_addr_df["customer_id"]).sum())

pd.notna(cust_addr_df["customer_id"].unique()).sum()

print("customer_ids not in demographics dataset:", sum([1 if (i not in cust_demo_df["customer_id"]) else 0 for i in cust_addr_df["customer_id"]]))

print("address blanks:", pd.isna(cust_addr_df["address"]).sum())
print("postcode blanks:", pd.isna(cust_addr_df["postcode"]).sum())

print("state:", cust_addr_df["state"].unique())
print("blanks:", pd.isna(cust_addr_df["state"]).sum())
plt.hist(cust_addr_df["state"][pd.notna(cust_addr_df["state"])], bins=5, edgecolor="black", align="mid")
plt.show()

print("country:", cust_addr_df["country"].unique())
print("blanks:", pd.isna(cust_addr_df["country"]).sum())

cust_addr_df["property_valuation"].describe()
print("property_valuation blanks:", pd.isna(cust_addr_df["property_valuation"]).sum())
plt.hist(cust_addr_df["property_valuation"], bins=10, edgecolor="black", align="mid")
plt.show()

###########################Transactions########################

transaction_df = pd.read_excel(data, header=1, sheet_name="Transactions")
transaction_df["transaction_date"].describe()
analysis_date = pd.to_datetime("2017-12-30")
transaction_df = transaction_df[transaction_df["transaction_date"] > (analysis_date - pd.DateOffset(months=3))]
transaction_df.head()

transaction_df.shape

duplicated_rows = transaction_df[transaction_df.duplicated()]
total_duplicate_rows = duplicated_rows.shape[0]

transaction_df["list_price"].describe()
print("list_price blanks:", pd.isna(transaction_df["list_price"]).sum())

print("standart_coats blanks:", pd.isna(transaction_df["standard_cost"]).sum())

transaction_df["profit"] = transaction_df["list_price"] - transaction_df["standard_cost"]
transaction_df["profit"].describe()

print("transaction_id blanks:", pd.isna(transaction_df["transaction_id"]).sum())
print("customer_id blanks:", pd.isna(transaction_df["customer_id"]).sum())
pd.notna(transaction_df["customer_id"].unique()).sum()

print("customer_ids not in demographic dataset:", sum([(1 if (i not in cust_demo_df["customer_id"]) else 0) for i in cust_addr_df["customer_id"]]))

print("customer_ids not in addresses dataset:", sum([(1 if (i not in cust_demo_df["customer_id"]) else 0) for i in transaction_df["customer_id"]]))

transaction_df["transaction_date"].describe()
print("transaction_date blanks:", pd.isna(transaction_df["transaction_date"]).sum())

print("online_order:", transaction_df["online_order"].unique())
print("blanks:", pd.isna(transaction_df["online_order"]).sum())
plt.hist(transaction_df["online_order"][pd.notna(transaction_df["online_order"])], bins=2, edgecolor="black", align="mid", color="tab:blue")
plt.xticks([0,1], ["N", "Y"])
plt.xlabel("Online Order")
plt.ylabel("Frequency")
plt.show()

transaction_df["order_status"][transaction_df["order_status"] == "Cancelled"].count()

print("brand:", transaction_df["brand"].unique())
print("blanks:", pd.isna(transaction_df["brand"]).sum())
plt.hist(transaction_df["brand"][pd.notna(transaction_df["brand"])], bins=5, edgecolor="black", align="mid")
plt.xlabel("Brands")
plt.ylabel("Quantity")
plt.show()

print("product_line:", transaction_df["product_line"].unique())
print("blanks:", pd.isna(transaction_df["product_line"]).sum())
plt.hist(transaction_df["product_line"][pd.notna(transaction_df["product_line"])], bins=4, edgecolor="black", align="mid")
plt.show()

print("product_class:", transaction_df["product_class"].unique())
print("blanks:", pd.isna(transaction_df["product_class"]).sum())
plt.hist(transaction_df["product_class"][pd.notna(transaction_df["product_class"])], bins=3, edgecolor="black", align="mid")
plt.show()

print("product_size", transaction_df["product_size"].unique())
print("blanks:", pd.isna(transaction_df["product_size"]).sum())
plt.hist(transaction_df["product_size"][pd.notna(transaction_df["product_size"])], bins=3, edgecolor="black", align="mid")
plt.show()

print("production_first_sold_date blanks:", pd.isna(transaction_df["product_first_sold_date"]).sum())
plt.scatter(transaction_df["brand"][pd.notna(transaction_df["brand"])], transaction_df["profit"][pd.notna(transaction_df["brand"])])
plt.show()

