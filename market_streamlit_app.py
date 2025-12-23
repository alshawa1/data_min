import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

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

st.title("🛒 Market Segmentation & Analysis")
st.write("### Using K-Means Clustering for Customer Segmentation")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    import os
    # Loading the dataset from the local directory
    # Use absolute path relative to this script to avoid CWD issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'online_retail_II.csv')
    
    if not os.path.exists(file_path):
        st.error(f"❌ File not found at: {file_path}")
        st.warning("Debugging Information:")
        st.write(f"**Script Directory:** `{script_dir}`")
        st.write(f"**Current Working Directory:** `{os.getcwd()}`")
        st.write("**Files in Script Directory:**")
        try:
            files = os.listdir(script_dir)
            st.code("\n".join(files))
        except Exception as e:
            st.error(f"Could not list files: {e}")
        return None

    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

df = load_data()

if df is not None:
    # --- DATA CLEANING ---
    st.sidebar.header("Data Processing")
    
    with st.spinner('Cleaning data...'):
        # Drop missing Customer ID
        df.dropna(subset=['Customer ID'], inplace=True)
        
        # Rename Invoice column to standard InvoiceNo if needed, though usually handled
        df.rename(columns={'ï»¿Invoice': 'InvoiceNo', 'Invoice': 'InvoiceNo'}, inplace=True)
        
        # Remove cancelled transactions
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
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # Elbow Method (Optional - pre-calculated or interactive)
    if st.checkbox("Show Elbow Plot (optimal K search)"):
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(rfm_scaled)
            wcss.append(kmeans.inertia_)
            
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

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
    }).sort_values(by='Monetary', ascending=True) # sort for consistency often helps reading
    
    st.dataframe(avg_df.style.background_gradient(cmap='Greens'))
    
    # Visualizations
    st.write("### Cluster Visualizations")
    
    tab1, tab2 = st.tabs(["2D Scatter Plots", "3D Visualization"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', title='Recency vs Monetary', log_x=True, log_y=True)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(rfm, x='Frequency', y='Monetary', color='Cluster', title='Frequency vs Monetary', log_x=True, log_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        fig_3d = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster', opacity=0.7, log_x=True, log_y=True, log_z=True)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Download
    csv = rfm.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Segmentation Results CSV",
        csv,
        "customer_segmentation.csv",
        "text/csv",
        key='download-csv'
    )

    # --- MARKET BASKET ANALYSIS ---
    st.markdown("---")
    st.title("🛍️ Market Basket Analysis")
    st.write("### Discover which products are frequently bought together (Association Rules)")

    # Data Prep for Market Basket
    # Filter by Country for performance and relevance
    countries = df['Country'].unique()
    selected_country = st.selectbox("Select Country for Analysis (filtering allows for faster processing)", countries, index=list(countries).index('United Kingdom') if 'United Kingdom' in countries else 0)
    
    basket = (df[df['Country'] == selected_country]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))

    # One-hot encoding
    def encode_units(x):
        return 1 if x >= 1 else 0

    basket_encoded = basket.applymap(encode_units)
    
    # Apriori
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        
        st.write(f"Processing {basket_encoded.shape[0]} transactions for {selected_country}...")
        
        min_support = st.slider("Minimum Support (Frequency of itemset)", 0.01, 0.5, 0.02, 0.01)
        
        # Calculate frequent itemsets
        frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
        
        if not frequent_itemsets.empty:
            # Generate rules
            min_threshold = st.slider("Minimum Lift (Strength of association)", 1.0, 10.0, 1.0, 0.1)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
            
            if not rules.empty:
                st.subheader("🔗 Top Association Rules")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))
                
                # Product Recommender
                st.subheader("💡 Product Recommender")
                all_products = list(basket_encoded.columns)
                selected_product = st.selectbox("Select a Product to see recommendations:", all_products)
                
                # Find rules where selected product is in antecedents
                recommendations = rules[rules['antecedents'].apply(lambda x: selected_product in x)]
                
                if not recommendations.empty:
                    st.write(f"**If a customer buys '{selected_product}', they are likely to buy:**")
                    st.table(recommendations[['consequents', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(5))
                else:
                    st.info(f"No strong recommendations found for {selected_product} with current thresholds.")
            else:
                st.warning("No rules found. Try lowering the thresholds.")
        else:
            st.warning("No frequent itemsets found. Try lowering the Minimum Support.")
            
    except ImportError:
        st.error("Please install mlxtend to use Market Basket Analysis: `pip install mlxtend`")

