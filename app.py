import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_style('whitegrid')
st.set_page_config(page_title='Titanic Survival Predictor', page_icon='🚢', layout='wide')

NUMERIC_FEATURES = ['Age', 'Fare', 'FamilySize', 'TicketGroupSize']
CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'IsAlone']
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DECK_OPTIONS = ['Unknown', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']


# ----------------------------------------------------------------------
# Copied verbatim from titanic.ipynb's engineer_features() cell
# ----------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Title from name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\.')
        title_map = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
            'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
            'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
        }
        df['Title'] = df['Title'].replace(title_map)
        df.loc[~df['Title'].isin(['Mr', 'Mrs', 'Miss', 'Master', 'Rare']), 'Title'] = 'Rare'
        df['Title'] = df['Title'].fillna('Rare')
    elif 'Title' not in df.columns:
        df['Title'] = 'Rare'

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Deck from cabin
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].str[0]
        df['Deck'] = df['Deck'].fillna('Unknown')
    elif 'Deck' not in df.columns:
        df['Deck'] = 'Unknown'

    # Ticket group size (people sharing the same ticket, incl. self)
    if 'Ticket' in df.columns and df['Ticket'].notna().any():
        ticket_counts = df['Ticket'].value_counts()
        df['TicketGroupSize'] = df['Ticket'].map(ticket_counts).fillna(1)
    elif 'TicketGroupSize' not in df.columns:
        df['TicketGroupSize'] = 1

    return df


@st.cache_resource
def load_model():
    # This is the exact object notebook trained: final_pipeline =
    # best_search.best_estimator_, refit on the full training set.
    return joblib.load('model.pkl')


@st.cache_data
def load_train_data():
    return pd.read_csv('train.csv').drop_duplicates()


model = load_model()
train_df = load_train_data()

st.title('🚢 Titanic Survival Predictor')
st.caption(
    "This app loads `model.pkl` — the exact pipeline your `titanic.ipynb` notebook "
    "trained and selected as best (GridSearchCV winner). Nothing is retrained here."
)

tab_predict, tab_batch, tab_eda = st.tabs(
    ['🔮 Predict', '📄 Batch Prediction (CSV)', '📊 EDA Dashboard']
)

# ----------------------------------------------------------------------
# TAB 1 — Single passenger prediction
# ----------------------------------------------------------------------
with tab_predict:
    st.subheader('Enter passenger details')

    col1, col2, col3 = st.columns(3)
    with col1:
        pclass = st.selectbox('Passenger Class', [1, 2, 3], index=2, help='1 = Upper, 2 = Middle, 3 = Lower')
        sex = st.selectbox('Sex', ['male', 'female'])
        age = st.slider('Age', 0, 80, 29)
    with col2:
        sibsp = st.number_input('Siblings / Spouses aboard', min_value=0, max_value=10, value=0)
        parch = st.number_input('Parents / Children aboard', min_value=0, max_value=10, value=0)
        fare = st.number_input('Fare paid ($)', min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    with col3:
        embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'],
                                 format_func=lambda x: {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}[x])
        title = st.selectbox('Title', ['Mr', 'Mrs', 'Miss', 'Master', 'Rare'])
        deck = st.selectbox('Deck (if known)', DECK_OPTIONS)

    if st.button('Predict Survival', type='primary'):
        passenger = pd.DataFrame([{
            'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch,
            'Fare': fare, 'Embarked': embarked, 'Title': title, 'Deck': deck
        }])
        passenger_fe = engineer_features(passenger)[FEATURE_COLS]

        pred = model.predict(passenger_fe)[0]
        proba = model.predict_proba(passenger_fe)[0]

        st.divider()
        result_col, gauge_col = st.columns([1, 2])
        with result_col:
            if pred == 1:
                st.success('### 🟢 Predicted: SURVIVED')
            else:
                st.error('### 🔴 Predicted: DID NOT SURVIVE')
            st.metric('Survival Probability', f'{proba[1]*100:.1f}%')
        with gauge_col:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh([0], [proba[1]], color='#27ae60', height=0.5)
            ax.barh([0], [proba[0]], left=[proba[1]], color='#c0392b', height=0.5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Probability')
            ax.text(proba[1] / 2, 0, f'{proba[1]*100:.0f}% survive', ha='center', va='center', color='white', fontweight='bold')
            if proba[0] > 0.08:
                ax.text(proba[1] + proba[0] / 2, 0, f'{proba[0]*100:.0f}% die', ha='center', va='center', color='white', fontweight='bold')
            st.pyplot(fig)

        st.caption(
            'This is a probabilistic estimate from a statistical model trained on historical data — '
            'not a certainty, and not a comment on any real individual.'
        )

# ----------------------------------------------------------------------
# TAB 2 — Batch prediction via CSV upload
# ----------------------------------------------------------------------
with tab_batch:
    st.subheader('Upload a CSV for batch predictions')
    st.caption(
        "Expected columns:"
        "PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked."
    )

    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_fe = engineer_features(batch_df)
            missing_cols = [c for c in FEATURE_COLS if c not in batch_fe.columns]
            if missing_cols:
                st.error(f'Missing required columns after processing: {missing_cols}')
            else:
                X_batch = batch_fe[FEATURE_COLS]
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)[:, 1]

                result_df = batch_df.copy()
                result_df['Survived_Prediction'] = preds
                result_df['Survival_Probability'] = probs.round(3)

                st.success(f'Predicted {len(result_df)} passengers.')
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions as CSV', csv_bytes,
                                    file_name='titanic_predictions.csv', mime='text/csv')
        except Exception as e:
            st.error(f'Could not process this file: {e}')

# ----------------------------------------------------------------------
# TAB 3 — EDA dashboard
# ----------------------------------------------------------------------
with tab_eda:
    st.subheader('Exploratory Data Analysis (training data)')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Passengers', len(train_df))
    m2.metric('Survival Rate', f"{train_df['Survived'].mean()*100:.1f}%")
    m3.metric('Median Age', f"{train_df['Age'].median():.0f}")
    m4.metric('Median Fare', f"${train_df['Fare'].median():.0f}")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=train_df, x='Sex', y='Survived', ax=ax, palette='Set2')
        ax.set_title('Survival Rate by Sex')
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=train_df, x='Pclass', y='Survived', ax=ax, palette='Set2')
        ax.set_title('Survival Rate by Class')
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=train_df, x='Age', hue='Survived', kde=True, bins=25,
                     ax=ax, palette=['#c0392b', '#27ae60'])
        ax.set_title('Age Distribution by Survival')
        st.pyplot(fig)
    with c4:
        fam = train_df.copy()
        fam['FamilySize'] = fam['SibSp'] + fam['Parch'] + 1
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=fam, x='FamilySize', y='Survived', ax=ax, palette='viridis')
        ax.set_title('Survival Rate by Family Size')
        st.pyplot(fig)

    st.subheader('Raw data sample')
    st.dataframe(train_df.head(20), use_container_width=True)
