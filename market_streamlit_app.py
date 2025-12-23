import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import os

# Set page config for a wider layout and better aesthetics
st.set_page_config(page_title="Market Segmentation Dashboard", layout="wide")

# Aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    h2 {
        color: #34495e;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'online_retail_II.zip')
    
    try:
        # Optimization: Read only necessary columns to save memory
        # 'Invoice' is sometimes 'InvoiceNo', handle variations by reading full then checking
        
        # Read FULL file then sample
        df = pd.read_csv(file_path, encoding='ISO-8859-1', compression='zip')
        
        # --- CRITICAL MEMORY FIX ---
        # Limit to 5,000 latest transactions for the free cloud tier
        # This is the "Nuclear Option" to absolutely prevent OOM
        if len(df) > 5000:
            st.toast("⚠️ Data is large! Sampling 5,000 rows for performance on free cloud.")
            df = df.sample(n=5000, random_state=42)
            
        return df
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Please ensure 'online_retail_II.zip' is uploaded.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    try:
        st.title("🛒 Market Segmentation & Analysis")
        st.write("### Using K-Means Clustering for Customer Segmentation")
        
        df = load_data()
        
        if df is None:
            return

        # --- DATA CLEANING ---
        st.sidebar.header("Data Processing")
        
        with st.spinner('Cleaning data...'):
            # Drop missing Customer ID
            df.dropna(subset=['Customer ID'], inplace=True)
            
            # Rename Invoice column to standard InvoiceNo if needed
            if 'Invoice' in df.columns:
                df.rename(columns={'Invoice': 'InvoiceNo'}, inplace=True)
            if 'ï»¿Invoice' in df.columns:
                df.rename(columns={'ï»¿Invoice': 'InvoiceNo'}, inplace=True)
            
            # Remove cancelled transactions
            # Ensure InvoicNo is string
            df['InvoiceNo'] = df['InvoiceNo'].astype(str)
            df = df[~df['InvoiceNo'].str.contains('C', na=False)]
            
            # Keep only positive quantity and price
            df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
            
            # Convert InvoiceDate
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            # Calculate Total Price
            df['Total_Price'] = df['Quantity'] * df['Price']

        st.success(f"Data Loaded & Cleaned! {df.shape[0]} transactions remaining.")

        # --- RFM CALCULATION ---
        st.subheader("📊 RFM Analysis")
        
        if df.empty:
            st.warning("No data available after cleaning.")
            return

        # Reference date (1 day after max date)
        ref_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
        
        # Aggregation
        rfm = df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (ref_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'Total_Price': 'sum'
        }).reset_index()
        
        rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", rfm['Customer ID'].nunique())
        col2.metric("Avg Recency", f"{rfm['Recency'].mean():.1f} days")
        col3.metric("Avg Monetary", f"${rfm['Monetary'].mean():.2f}")

        if st.checkbox("Show Raw RFM Data"):
            st.dataframe(rfm.head())

        # --- CLUSTERING ---
        st.subheader("🤖 K-Means Clustering")
        
        # Preprocessing for K-Means (Log transform to unclump data)
        # Handle zeros/negatives by using log1p
        rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
        
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)
        
        # Cluster Selection
        k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 3)
        
        kmeans = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(rfm_scaled)
        
        rfm['Cluster'] = clusters
        
        # Cluster Analysis
        st.write(f"### Segmentation Results (K={k_clusters})")
        
        # Summary Table
        avg_df = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary', 'Customer ID']].agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Customer ID': 'count'
        }).sort_values(by='Monetary', ascending=True) 
        
        st.dataframe(avg_df.style.background_gradient(cmap='Greens'))
        
        # Visualizations (2D ONLY for stability)
        st.write("### Cluster Visualizations (2D)")
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', title='Recency vs Monetary', log_x=True, log_y=True)
            st.plotly_chart(fig)
        with c2:
            fig = px.scatter(rfm, x='Frequency', y='Monetary', color='Cluster', title='Frequency vs Monetary', log_x=True, log_y=True)
            st.plotly_chart(fig)
        
        # --- MARKET BASKET ANALYSIS ---
        st.markdown("---")
        st.title("🛍️ Market Basket Analysis")
        st.write("### Discover which products are frequently bought together (Association Rules)")

        # Filter by Country for performance
        countries = df['Country'].unique()
        selected_country = st.selectbox("Select Country for Analysis", countries, index=list(countries).index('United Kingdom') if 'United Kingdom' in countries else 0)
        
        st.info("⚠️ Note: optimizing for performance. Analyzing top 100 most frequent items only.")

        # ONE-HOT ENCODING (Optimized)
        basket_subset = df[df['Country'] == selected_country]
        
        if basket_subset.empty:
            st.warning("No data for this country.")
        else:
            # 1. Aggressively limit to top 100 items by quantity to prevent OOM
            # Using size() instead of sum() of quantity is sometimes safer/faster for frequency
            top_items = basket_subset['Description'].value_counts().head(100).index
            basket_subset = basket_subset[basket_subset['Description'].isin(top_items)]

            # 2. Optimized Pivot
            basket = (basket_subset
                .groupby(['InvoiceNo', 'Description'])['Quantity']
                .sum().unstack().fillna(0))

            # 3. Convert to boolean
            basket_encoded = basket.apply(lambda x: x > 0)

            # Apriori
            from mlxtend.frequent_patterns import apriori, association_rules
            
            st.write(f"Processing {basket_encoded.shape[0]} transactions and {basket_encoded.shape[1]} items...")
            
            # Increase default min_support
            min_support = st.slider("Minimum Support", 0.01, 0.2, 0.02, 0.01)
            
            frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
            
            if not frequent_itemsets.empty:
                # Generate rules
                min_threshold = st.slider("Minimum Lift", 1.0, 10.0, 1.0, 0.1)
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
                rules = rules.replace([np.inf, -np.inf], np.nan).dropna()
                
                if not rules.empty:
                    st.subheader("🔗 Top Association Rules")
                    
                    # Make a copy for display
                    rules_display = rules.copy()
                    rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    st.dataframe(rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))
                    
                    # Product Recommender
                    st.subheader("💡 Product Recommender")
                    all_products = list(basket_encoded.columns)
                    selected_product = st.selectbox("Select a Product:", all_products)
                    
                    recommendations = rules[rules['antecedents'].apply(lambda x: selected_product in x)]
                    
                    if not recommendations.empty:
                        st.write(f"**If a customer buys '{selected_product}', they are likely to buy:**")
                        
                        recs_display = recommendations.copy()
                        recs_display['consequents'] = recs_display['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        st.table(recs_display[['consequents', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(5))
                    else:
                        st.info(f"No strong recommendations found for {selected_product} with current thresholds.")
                else:
                    st.warning("No rules found. Try lowering the thresholds.")
            else:
                st.warning("No frequent itemsets found. Try lowering the Minimum Support.")
                    
    except Exception as e:
        st.error("❌ An unexpected error occurred. Please refresh or contact support.")
        st.warning(f"Error Details (for debugging): {e}")

if __name__ == "__main__":
    main()
