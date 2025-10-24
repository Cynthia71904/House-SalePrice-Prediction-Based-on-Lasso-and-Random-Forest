import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['New roman']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning,module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning,module='seaborn')

df = pd.read_excel(r"C:\Users\Cynth\houpr_predi\price.xlsx")
df.head()

df = df.copy()
df = df.drop_duplicates()
df = df.dropna()
district_counts = df["MSZoning"].value_counts()
average_price_by_district = df.groupby("MSZoning")["SalePrice"].mean().sort_values(ascending=False)
area_distribution = df["LotArea"]
yearly_listings = df["YearRemodAdd"].value_counts().sort_index()
total_price_distribution = df["SalePrice"]

plt.figure(figsize=(10,6))
plt.hist(area_distribution, bins=30,edgecolor='black')
plt.title('Distribution of LotArea')
plt.xlabel('LotArea')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df["SalePrice"],bins=50,kde=True,color='red')
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel("Count")
plt.tight_layout()
plt.show()

correlation_features = df[["SalePrice","LotArea", "YearRemodAdd","OverallCond","TotalBsmtSF"]].corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_features,annot=True,cmap="coolwarm",fmt=".2f",linewidths=.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
