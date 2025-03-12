import xlrd
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

data = "dataset/KPMG_VI_New_raw_data_update_final.xlsx"
print(pd.ExcelFile(data).sheet_names)


###################Customer Demographic###################
ddf = pd.read_excel(data, header=1, sheet_name="CustomerDemographic", index_col=0)
ddf.head()

ddf = ddf.drop(["first_name", "last_name", "default", "job_title"], axis=1)

def replace_values(df, column, mapping):
    """
    Replaces values in a specified column of a DataFrame with the given mapping.

    Parameters:
    - df: DataFrame, the dataset to operate on
    - column: str, the name of the column to be replaced
    - mapping: dict, a dictionary mapping old values to new values
    """
    df[column].replace(mapping, inplace=True)

replace_values(ddf, 'gender', {'F': 'Female', 'Femal': 'Female', 'M': 'Male', 'U': 'Unknown'})
replace_values(ddf, 'deceased_indicator', {'N': 0, 'Y': 1})
replace_values(ddf, 'owns_car', {'Yes': 1, 'No': 0})

print(ddf)

ddf["DOB"] = pd.to_datetime(ddf["DOB"])
ddf = ddf[ddf["DOB"] != ddf.DOB.min()]

ddf[ddf["deceased_indicator"] == 0]
ddf = ddf.drop(["deceased_indicator"], axis=1)
ddf_cleaned = ddf.dropna()
ddf_cleaned.head()

ddf_cleaned.isnull().sum()
ddf_cleaned["age"] = (dt.datetime.now() - ddf_cleaned["DOB"]) / np.timedelta64(1,"Y")
ddf_cleaned["age_class"] = ((round(ddf_cleaned["age"] / 10)).astype(int))
ddf_cleaned.head()
# ddf_cleaned["age_class"].nunique()
# ddf_cleaned["age_class"].unique()
# ddf_cleaned["age_class"].value_counts().idxmax()
# max_purchased_class = ddf_cleaned[ddf_cleaned["age_class"] == 5]["age"]
# max_purchased_class.describe()

# ddf_cleaned["owns_car"].value_counts()

females_df = ddf_cleaned[ddf_cleaned["gender"] == "Female"]

age_class_counts = females_df["age_class"].value_counts()

max_purchased_females_class = age_class_counts.idxmax()

print(f"Kadın müşterilerin en yoğun olduğu yaş grubu: {max_purchased_females_class}")

############Customer Address#################
addf = pd.read_excel(data, header=1, sheet_name="CustomerAddress", index_col=0)
addf.head()

replace_values(addf, "state", {"New South Wales": "NSW"})
replace_values(addf, "state", {"Victoria": "VIC"})

addf_cleaned = addf.dropna()
addf_cleaned.isnull().sum()

ddf_addf = pd.merge(ddf_cleaned, addf_cleaned, left_index=True, right_index=True)
ddf_addf = ddf_addf.dropna()
ddf_addf.head()

##################Transactions######################

transaction_df = pd.read_excel(data, header=1, sheet_name="Transactions", index_col=0)
transaction_df = transaction_df.sort_values("customer_id")
transaction_df.head()

transaction_df["order_status"].value_counts()
most_cancelled_brand = transaction_df[transaction_df["order_status"] == "Cancelled"]["brand"].value_counts().idxmax()

transaction_df.isna().sum()
transaction_df.duplicated().sum()
transaction_df.shape

analysis_date = pd.to_datetime("2017-12-30")
transaction_df = transaction_df[transaction_df["transaction_date"] > (analysis_date - pd.DateOffset(months=3))]
transaction_df.head()

transaction_df['product_first_sold_date'] = pd.TimedeltaIndex(transaction_df['product_first_sold_date'], unit="d") + dt.datetime(1900,1,1)
transaction_df.head()

transaction_df.shape

transaction_df_clean = transaction_df.dropna()
transaction_df_clean.shape

transaction_df_clean["transaction_date"].describe()

most_recent_purchase = transaction_df_clean["transaction_date"].max()
transaction_df_clean["last_purchase_days_ago"] = most_recent_purchase - transaction_df_clean["transaction_date"]
transaction_df_clean["last_purchase_days_ago"] /= np.timedelta64(1, "D")
transaction_df_clean["profit"] = transaction_df_clean["list_price"] - transaction_df_clean["standard_cost"]
transaction_df_clean.head()

transaction_df_clean[transaction_df_clean["last_purchase_days_ago"] > 365].shape

rfm = transaction_df_clean.groupby("customer_id").agg({"last_purchase_days_ago": lambda x: x.min(),
                                                       "customer_id": lambda x: len(x),
                                                       "profit": lambda x: x.sum()
})

rfm.rename(columns={"last_purchase_days_ago": "recency",
"customer_id": "frequency",
"profit": "monetary"}, inplace=True
)

rfm.shape

quartiles = rfm.quantile(q=[0.25,0.50,0.75])
quartiles


def ROneHotEncoder(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.5]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4


def FMOneHotEncoder(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.5]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 1

rfmSeg = rfm
rfmSeg["r_score"] = rfmSeg["recency"].apply(ROneHotEncoder, args=("recency", quartiles))
rfmSeg["f_score"] = rfmSeg["frequency"].apply(FMOneHotEncoder, args=("frequency", quartiles))
rfmSeg["m_score"] = rfmSeg["monetary"].apply(FMOneHotEncoder, args=("monetary", quartiles))

rfmSeg.head()

rfmSeg["rfm_class"] = 100 * rfmSeg["r_score"] + 10 * rfmSeg["f_score"] + rfmSeg["m_score"]
rfmSeg["total_score"] = rfmSeg["r_score"] + rfmSeg["f_score"] + rfmSeg["m_score"]
rfmSeg.head()

rfm_quartiles = (rfmSeg["rfm_class"].min(), rfmSeg["rfm_class"].quantile(q=0.25),
                 rfmSeg["rfm_class"].median(), rfmSeg["rfm_class"].quantile(q=0.75),
                 rfmSeg["rfm_class"].max())

rfm_quartiles

def RFMClassOneHotEncoder(x, p, d):
    if x <= d[0]:
        return 'gold'
    elif x <= d[1]:
        return 'silver'
    elif x <= d[2]:
        return 'bronze'
    else:
        return 'basic'

rfmSeg['customer_title'] = rfmSeg['rfm_class'].apply(RFMClassOneHotEncoder, args=('rfm_class', rfm_quartiles))
rfmSeg