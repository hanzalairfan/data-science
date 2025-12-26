import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Page Configuration & Styling
# =========================
st.set_page_config(page_title="🎮 Video Game Sales Prediction", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content container */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Title styling */
    h1 {
        color: #667eea;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #764ba2;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Title and Description
# =========================
st.title("🎮 Video Game Sales Classification")
st.markdown("<p style='text-align: center; color: #666; font-size: 18px;'>Predict <strong>Sales Category (Low / Medium / High)</strong> using Machine Learning models</p>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# Load dataset
# =========================
df = pd.read_csv("vgsales.csv")

# Handle missing values
df['Publisher'] = df['Publisher'].fillna('Unknown')
df = df.dropna(subset=["Year", "Genre", "Platform"])

# =========================
# Create Sales Classes
# =========================
df["Sales_Class"] = pd.cut(
    df["Global_Sales"],
    bins=[-1, 1, 10, df["Global_Sales"].max()],
    labels=["Low", "Medium", "High"]
)

# =========================
# Encode categorical columns
# =========================
encoders = {}
for col in ['Platform', 'Genre', 'Publisher']:
    encoders[col] = LabelEncoder()
    df[col + '_Encoded'] = encoders[col].fit_transform(df[col])

# =========================
# Features & Target
# =========================
X = df[[
    "Year",
    "Platform_Encoded",
    "Genre_Encoded",
    "Publisher_Encoded",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales"
]]

# Rename columns to remove _Encoded suffix for consistency
X.columns = ["Year", "Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

y = df["Sales_Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Train ALL models and store metrics
# =========================
if not os.path.exists("models"):
    os.makedirs("models")

models_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

# Train and save models
for model_name, model in models_dict.items():
    model_path = f"models/{model_name.lower().replace(' ', '_')}.pkl"
    if not os.path.exists(model_path):
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, model_path)

# Save scaler and encoders
if not os.path.exists("models/scaler.pkl"):
    joblib.dump(scaler, "models/scaler.pkl")
if not os.path.exists("models/encoders.pkl"):
    joblib.dump(encoders, "models/encoders.pkl")

# Load encoders
encoders = joblib.load("models/encoders.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# Function to calculate all metrics for a model
# =========================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        metrics["AUC"] = roc_auc_score(
            pd.get_dummies(y_test), y_proba, multi_class="ovr"
        )
    else:
        metrics["AUC"] = None
    
    return metrics, y_pred

# =========================
# Create Tabs for Different Views
# =========================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Comparison", "🌲 Feature Analysis"])

# =========================
# TAB 1: PREDICTION
# =========================
with tab1:
    # Model selection
    col1, col2 = st.columns([2, 1])
    with col1:
        model_name = st.selectbox("🤖 Select Machine Learning Model", 
                                  ["Logistic Regression", "Decision Tree", "Random Forest"])

    model = joblib.load(f"models/{model_name.lower().replace(' ', '_')}.pkl")

    st.markdown("---")

    # Input Section
    st.subheader("📊 Enter Game Details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        year = st.number_input("📅 Year", min_value=1980, max_value=2025, value=2010)
        platform_options = sorted(df['Platform'].unique())
        platform = st.selectbox("🎮 Platform", options=platform_options)

    with col2:
        genre_options = sorted(df['Genre'].unique())
        genre = st.selectbox("🎯 Genre", options=genre_options)
        publisher_options = sorted(df['Publisher'].unique())
        publisher = st.selectbox("🏢 Publisher", options=publisher_options)

    with col3:
        na_sales = st.number_input("🇺🇸 NA Sales (millions)", min_value=0.0, step=0.1, value=0.5)
        eu_sales = st.number_input("🇪🇺 EU Sales (millions)", min_value=0.0, step=0.1, value=0.3)

    with col4:
        jp_sales = st.number_input("🇯🇵 JP Sales (millions)", min_value=0.0, step=0.1, value=0.2)
        other_sales = st.number_input("🌍 Other Sales (millions)", min_value=0.0, step=0.1, value=0.1)

    st.markdown("---")

    # Action Buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        predict_button = st.button("🔮 Predict Sales Category")

    with col2:
        clear_button = st.button("🗑️ Clear Output")

    if clear_button:
        st.rerun()

    # Prediction
    if predict_button:
        # Encode categorical inputs
        platform_encoded = encoders['Platform'].transform([platform])[0]
        genre_encoded = encoders['Genre'].transform([genre])[0]
        publisher_encoded = encoders['Publisher'].transform([publisher])[0]
        
        input_data = pd.DataFrame([[year, platform_encoded, genre_encoded, publisher_encoded, 
                                    na_sales, eu_sales, jp_sales, other_sales]],
                                 columns=["Year", "Platform", "Genre", "Publisher", 
                                         "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"])
        input_scaled = scaler.transform(input_data)
        
        # Get prediction
        prediction = model.predict(input_scaled)
        
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        # Display prediction with styled box
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(input_scaled)[0]
            
            result_col, chart_col = st.columns([1, 2])
            
            with result_col:
                color_map = {"Low": "#ff6b6b", "Medium": "#4ecdc4", "High": "#95e1d3"}
                pred_color = color_map.get(prediction[0], "#667eea")
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {pred_color} 0%, {pred_color}dd 100%); 
                            padding: 2rem; border-radius: 15px; text-align: center; 
                            box-shadow: 0 8px 20px rgba(0,0,0,0.15);'>
                    <h2 style='color: white; margin: 0; font-size: 24px;'>Predicted Category</h2>
                    <h1 style='color: white; margin: 10px 0 0 0; font-size: 48px; font-weight: 800;'>{prediction[0]}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("**📊 Confidence Levels:**")
                for category, prob in zip(["Low", "Medium", "High"], probas):
                    st.markdown(f"**{category}:** {prob*100:.1f}%")
            
            with chart_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sales_categories = ["Low", "Medium", "High"]
                colors = ["#ff6b6b", "#4ecdc4", "#95e1d3"]
                
                bars = ax.barh(sales_categories, probas, color=colors, height=0.6)
                
                for bar, prob in zip(bars, probas):
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                           f'{prob*100:.1f}%', 
                           ha='left', va='center', 
                           fontsize=12, fontweight='bold',
                           color='#333333')
                
                ax.set_xlabel("Probability", fontsize=12, fontweight='bold', color='#333')
                ax.set_ylabel("Sales Category", fontsize=12, fontweight='bold', color='#333')
                ax.set_title("Prediction Probability Distribution", 
                            fontsize=14, fontweight='bold', color='#667eea', pad=20)
                ax.set_xlim(0, 1.15)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#cccccc')
                ax.spines['bottom'].set_color('#cccccc')
                ax.tick_params(colors='#666666')
                ax.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('white')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
        else:
            st.success(f"✅ Predicted Sales Category: **{prediction[0]}**")

# =========================
# TAB 2: MODEL COMPARISON
# =========================
with tab2:
    st.subheader("📊 Comprehensive Model Comparison")
    st.markdown("Compare performance metrics across all trained models")
    
    # Evaluate all models
    all_metrics = {}
    all_predictions = {}
    
    for model_name in models_dict.keys():
        model = joblib.load(f"models/{model_name.lower().replace(' ', '_')}.pkl")
        metrics, y_pred = evaluate_model(model, X_test_scaled, y_test)
        all_metrics[model_name] = metrics
        all_predictions[model_name] = y_pred
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(all_metrics).T
    
    # Display metrics table
    st.markdown("### 📋 Metrics Summary Table")
    styled_df = comparison_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0)
    st.dataframe(styled_df, use_container_width=True)
    
    # Find best model for each metric
    st.markdown("### 🏆 Best Performing Models")
    col1, col2, col3 = st.columns(3)
    
    metrics_to_show = ['Accuracy', 'F1-Score', 'AUC']
    cols = [col1, col2, col3]
    
    for metric, col in zip(metrics_to_show, cols):
        if metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            
            with col:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 10px; text-align: center;'>
                    <h4 style='color: white; margin: 0; font-size: 16px;'>{metric}</h4>
                    <h3 style='color: white; margin: 5px 0; font-size: 20px;'>{best_model}</h3>
                    <p style='color: white; margin: 0; font-size: 24px; font-weight: bold;'>{best_score:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization options
    st.markdown("### 📈 Visual Comparison")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Grouped bar chart for all metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        comparison_plot = comparison_df.drop(columns=['AUC']) if 'AUC' in comparison_df.columns else comparison_df
        comparison_plot.plot(kind='bar', ax=ax, width=0.8, colormap='viridis')
        
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold', color='#764ba2')
        ax.set_xlabel("Model", fontsize=12, fontweight='bold')
        ax.set_ylabel("Score", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with viz_col2:
        # Radar chart for model comparison
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = ['#667eea', '#764ba2', '#ff6b6b']
        
        for (model_name, metrics), color in zip(all_metrics.items(), colors):
            values = [metrics[m] for m in metrics_for_radar]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_for_radar, size=10)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Radar Chart", size=14, fontweight='bold', 
                     color='#764ba2', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Confusion matrices
    st.markdown("### 🎯 Confusion Matrices")
    
    cm_cols = st.columns(3)
    
    for idx, (model_name, y_pred) in enumerate(all_predictions.items()):
        with cm_cols[idx]:
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Low', 'Medium', 'High'],
                       yticklabels=['Low', 'Medium', 'High'],
                       cbar_kws={'label': 'Count'})
            
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', color='#764ba2')
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# =========================
# TAB 3: FEATURE ANALYSIS
# =========================
with tab3:
    st.subheader("🌲 Random Forest Feature Importance Analysis")
    st.markdown("Understanding which features contribute most to predictions")
    
    # Load Random Forest model
    rf_model = joblib.load("models/random_forest.pkl")
    
    # Get feature importances
    importances = rf_model.feature_importances_
    feature_names = ["Year", "Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    
    # Create dataframe
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance_sorted = importance_df.sort_values(by="Importance", ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_sorted)))
        
        bars = ax.barh(importance_sorted["Feature"], importance_sorted["Importance"], color=colors)
        
        ax.set_title("Feature Importance Ranking", fontsize=14, fontweight='bold', color='#764ba2')
        ax.set_xlabel("Importance Score", fontsize=12, fontweight='bold')
        ax.set_ylabel("Feature", fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', 
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(importance_df)))
        
        wedges, texts, autotexts = ax.pie(
            importance_df["Importance"], 
            labels=importance_df["Feature"],
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
        
        ax.set_title("Feature Importance Distribution", 
                     fontsize=14, fontweight='bold', color='#764ba2', pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Feature importance table
    st.markdown("### 📋 Detailed Feature Importance")
    
    # Calculate percentage
    importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
    importance_df['Cumulative %'] = importance_df['Percentage'].cumsum()
    
    # Display styled table
    styled_importance = importance_df.style.format({
        'Importance': '{:.6f}',
        'Percentage': '{:.2f}%',
        'Cumulative %': '{:.2f}%'
    }).background_gradient(subset=['Importance'], cmap='YlGnBu')
    
    st.dataframe(styled_importance, use_container_width=True)
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    
    top_feature = importance_df.iloc[0]['Feature']
    top_importance = importance_df.iloc[0]['Percentage']
    
    top3_cumulative = importance_df.head(3)['Percentage'].sum()
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.info(f"**Most Important Feature:** {top_feature} contributes {top_importance:.1f}% to the model's predictions")
    
    with insight_col2:
        st.info(f"**Top 3 Features:** Account for {top3_cumulative:.1f}% of total importance")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 14px;'>Built with ❤️ using Streamlit & Scikit-learn</p>", unsafe_allow_html=True)