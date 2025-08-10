# 🍗 Wings R Us - Smart Recommendation Engine

## 🚀 Project Overview
This project addresses a business problem for **Wings R Us**, a US-based Quick Service Restaurant (QSR) chain. The goal is to design and implement a smart recommendation system to enhance the digital checkout experience, with the primary objective of **increasing the Average Order Value (AOV)**.

The solution involves a **hybrid, context-aware recommendation engine** that provides different types of suggestions based on the customer segment. This repository contains the complete workflow, from initial data cleaning and exploratory data analysis (EDA) to model training and a fully functional **Streamlit web application** for demonstration. The app also integrates the **Gemini API** to provide natural language justifications for its recommendations.

---

## 🌍 Live Resources
- **Streamlit App** → [Wings R Us Recommender System](https://green-analysts-wings-r-us-recommender-system.streamlit.app/)
- **Power BI Dashboard** → [Download WWT_dashboard.pbix](./WWT_dashboard.pbix) *(Requires Power BI Desktop)*

---

## 📂 File Structure

| File | Description |
|------|-------------|
| `Data_Cleaning.ipynb` | End-to-end data cleaning: handling missing values, parsing nested JSON, merging datasets |
| `EDA.ipynb` | Exploratory Data Analysis, customer segmentation (RFM), business insights |
| `Recommendation.ipynb` | Trains FP-Growth & Collaborative Filtering models, exports them as `.joblib` |
| `App.py` | Streamlit app loading models and providing the recommendation UI |
| `guest_fp_growth_model.joblib` | Pre-trained FP-Growth model for Guest users |
| `loyal_collab_filtering_model.joblib` | Pre-trained Collaborative Filtering model for Registered/eClub users |
| `requirements.txt` | Python dependencies |
| `.env` | Stores your Gemini API key (not committed to GitHub) |

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variable

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

---

## ▶️ How to Run

### 1. Data Processing and Model Training (Optional)
Pre-trained models (`.joblib`) are already included. To retrain:

- Run `Data_Cleaning.ipynb` → outputs `final_merged_data.csv`
- Run `EDA.ipynb` → performs segmentation, insights
- Run `Recommendation.ipynb` → saves model files

### 2. Launch Streamlit Application
```bash
streamlit run App.py
```

This will open a browser tab with the interactive app.

---

## 🤖 Modeling Approach

### Guest Customers → FP-Growth (Market Basket Analysis)

- **What it is**: Finds frequently co-purchased items across thousands of orders.
- **Why**: Effective for guest users without historical data.

📌 *Example:*  
*“Customers who buy spicy wings also buy ranch dip.”*

---

### Registered/eClub Customers → Collaborative Filtering

- **What it is**: Personalized recommendations based on item-item similarity from purchase history.
- **Why**: Better for loyal users with identifiable patterns.

📌 *Example:*  
*“Since you often buy spicy items, you might like this new spicy flavor you haven't tried yet.”*

---

## 📊 Key Insights from EDA

- **AOV Paradox**: Loyal customers have lower AOV ($39.29) than Guest users ($53.15).
- **Two-Speed Customer Base**: A mix of high-value new users and loyal regulars → supports hybrid model.
- **Geographic Concentration**: Heavy Texas presence suggests future location-aware optimization.

---

## 📝 License
This project is for academic/demo purposes. Adapt and use freely.

---

## 🙌 Credits
Built using Python, Streamlit, and Gemini AI ✨