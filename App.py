import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import fpgrowth, association_rules
# from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import cosine_similarity
import joblib
import google.generativeai as genai
import gdown

# Direct Google Drive download links
GUEST_CSV_URL = "https://drive.google.com/uc?export=download&id=1uB2Fzzz4-Hu1GO1n0pfuMjN1iORL1KZz"
LOYAL_CSV_URL = "https://drive.google.com/uc?export=download&id=1-IYcACfShaVzVOsQNFUaeejrL_uLcV7L"

def download_if_missing():
    """Download CSVs from Google Drive if not present locally."""
    if not os.path.exists("guest_customer_data.csv"):
        print("Downloading guest_customer_data.csv...")
        gdown.download(GUEST_CSV_URL, "guest_customer_data.csv", quiet=False)

    if not os.path.exists("loyal_customer_data.csv"):
        print("Downloading loyal_customer_data.csv...")
        gdown.download(LOYAL_CSV_URL, "loyal_customer_data.csv", quiet=False)

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

def generate_recommendation_justification(cart_items, recommendations):
    prompt = f"""
You are a recommendation engine assistant for a quick-service restaurant.

The customer has selected the following items in their cart: {', '.join(cart_items)}.

Based on this, the system recommends: {', '.join(recommendations)}.

Explain **why** these recommendations might make sense from a food pairing or upsell perspective.
Keep the explanation short and easy to understand (2-3 sentences).
    """

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()

# --- 1. Caching and Model Loading ---
# This section is now more memory-efficient by loading pre-processed data.
@st.cache_data
def load_and_prepare_data():
    """
    Loads pre-processed datasets and prepares lists for the UI.
    This is much more memory-efficient as it avoids loading the entire raw dataset.
    """
    # NOTE: You must create these smaller CSV files first.
    # See instructions on how to create 'loyal_customer_data.csv' and 'guest_customer_data.csv'.

    download_if_missing()  # Ensure CSVs are available
    
    try:
        df_loyal = pd.read_csv('loyal_customer_data.csv')
        df_guest = pd.read_csv('guest_customer_data.csv')
    except FileNotFoundError:
        st.error("Error: Make sure 'loyal_customer_data.csv' and 'guest_customer_data.csv' are in the same directory.")
        return None, None, [], []

    # Get unique lists for UI selectors from both datasets
    all_items = sorted(pd.concat([df_loyal['item_name'], df_guest['item_name']]).unique())
    all_stores = sorted(pd.concat([df_loyal['STORE_NUMBER'], df_guest['STORE_NUMBER']]).unique())
    
    return df_loyal, df_guest, all_items, all_stores

@st.cache_resource
def load_guest_model(df_guest):
    """
    Loads the pre-trained FP-Growth model from a joblib file.
    """
    try:
        # The filename should match what you used when you dumped the model
        return joblib.load('guest_fp_growth_model.joblib')
    except FileNotFoundError:
        st.error("Error: 'guest_fp_growth_model.joblib' not found. Please run your training notebook to create it.")
        return None

@st.cache_resource
def load_loyal_model(df):
    """
    Loads the pre-trained Collaborative Filtering model from a joblib file.
    """
    try:
        # The filename should match what you used when you dumped the model
        return joblib.load('loyal_collab_filtering_model.joblib')
    except FileNotFoundError:
        st.error("Error: 'loyal_collab_filtering_model.joblib' not found. Please run your training notebook to create it.")
        return None

# --- 2. Recommendation Functions (Unchanged) ---
def get_guest_recommendations(cart_items, rules_df):
    matching_rules = rules_df[rules_df['antecedents'].apply(lambda x: x.issubset(cart_items))]
    if matching_rules.empty: return ["N/A", "N/A", "N/A"]
    sorted_rules = matching_rules.sort_values(by='lift', ascending=False)
    recommendations = sorted_rules['consequents'].explode().unique()
    final_recs = [rec for rec in recommendations if rec not in cart_items]
    while len(final_recs) < 3: final_recs.append("N/A")
    return final_recs[:3]

def get_personalized_recommendations(cart_items, similarity_df):
    similar_scores = pd.Series(dtype=float)
    for item in cart_items:
        if item in similarity_df.columns:
            similar_scores = pd.concat([similar_scores, similarity_df[item]])
    if similar_scores.empty: return ["N/A", "N/A", "N/A"]
    grouped_scores = similar_scores.groupby(similar_scores.index).mean()
    sorted_scores = grouped_scores.sort_values(ascending=False)
    recommendations = [item for item in sorted_scores.index if item not in cart_items]
    while len(recommendations) < 3: recommendations.append("N/A")
    return recommendations[:3]

# --- 3. Streamlit App UI and Logic ---
st.set_page_config(layout="wide", page_title="Wings R Us Recommender")

st.title(" Wings R Us Smart Recommender")
st.markdown("Select customer and order details to get personalized item recommendations.")

# Load data and models
df_loyal, df_guest, all_items, all_stores = load_and_prepare_data()

# Check if data loading was successful before proceeding
if df_loyal is not None and df_guest is not None:
    rules_guest = load_guest_model(df_guest)
    item_similarity_df_loyal = load_loyal_model(df_loyal)

    st.sidebar.header("Order Details")

    # User selection widgets in the sidebar
    customer_type = st.sidebar.selectbox(
        "Select Customer Type",
        ('Guest', 'Registered', 'eClub')
    )

    store_number = st.sidebar.selectbox(
        "Select Store Number",
        all_stores
    )

    occasion_type = st.sidebar.selectbox(
        "Select Order Occasion",
        ('ToGo', 'Delivery')
    )

    st.sidebar.header("Build Your Cart")

    # Item selection with logic to prevent duplicates
    item1 = st.sidebar.selectbox("Select Item 1", ["-"] + all_items)

    items_for_2 = ["-"] + [item for item in all_items if item != item1]
    item2 = st.sidebar.selectbox("Select Item 2", items_for_2)

    items_for_3 = ["-"] + [item for item in all_items if item not in [item1, item2]]
    item3 = st.sidebar.selectbox("Select Item 3", items_for_3)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Cart")
        cart_items = [item for item in [item1, item2, item3] if item != "-"]
        if not cart_items:
            st.warning("Please select at least one item to get recommendations.")
        else:
            for item in cart_items:
                st.info(item)

    # Recommendation logic
    if st.sidebar.button("Get Recommendations", use_container_width=True):
        if not cart_items:
            st.error("You must select at least one item in your cart.")
        else:
            recommendations = []
            if customer_type == 'Guest':
                st.success("Using Guest Model (FP-Growth)")
                recommendations = get_guest_recommendations(set(cart_items), rules_guest)
            else: # Registered or eClub
                st.success("Using Loyal Customer Model (Collaborative Filtering)")
                recommendations = get_personalized_recommendations(set(cart_items), item_similarity_df_loyal)
            
            with col2:
                st.subheader("Here are your Top 3 Recommendations:")
                for rec in recommendations:
                    st.info(f"**{rec}**")

            # Line Break
            st.markdown("---")

            # Gemini Justification
            st.subheader("🤖 Gemini Justification:")
            with st.spinner("Generating explanation..."):
                explanation = generate_recommendation_justification(cart_items, recommendations)
                st.info(explanation)