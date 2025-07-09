# Importing Required Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
import time as t
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Loading Titanic Dataset
data=sns.load_dataset('titanic')

# Filling Missing Values
data['age'].fillna(data['age'].median(),inplace=True)
data['embarked']=LabelEncoder().fit_transform(data['embarked'])
data['sex']=LabelEncoder().fit_transform(data['sex'])

# print(data.isnull().sum())

# Features And Target
features=data[['age','pclass','sex','sibsp','parch','fare','embarked']]
x , y = features , data['survived']

# Splting Data Into 80 - 20 To Train And Test
x_train , x_test , y_train , y_test = train_test_split( x , y , test_size=0.2 , random_state=42)

# Initializing Classifiers
model1=RandomForestClassifier(n_estimators=5)
model2=KNeighborsClassifier(n_neighbors=5)
model3=LogisticRegression(max_iter=1000)

# Fitting Models
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)


st.set_page_config(page_title="Titanic Surviver Finder",page_icon="cruise.png")

st.sidebar.title("Choose Classifier")
classifier=st.sidebar.selectbox("",['RandomForestClassifier','KNN','LogisticRegression'])
ex=st.sidebar.checkbox("Explain Classifiers")
compare = st.sidebar.checkbox("Compare All Models")
raw=st.sidebar.checkbox("Raw Data Of Titanic")
st.header("Titanic Surviver Founder")

if ex:
    with st.expander("🧠 How Classifiers Work",expanded=True):
        st.markdown("""
        - **Random Forest**: Ensemble of decision trees that vote on the final outcome.
        - **Logistic Regression**: A statistical model estimating the probability of survival.
        - **KNN**: Looks at similar passengers and classifies based on neighbors.
        """)     

# User Inputs
with st.expander("**User Data**",expanded=True):  
    col1,col2=st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3])
        sex = st.radio("Sex", options=["male", "female"])
        sexx= 0 if sex=='male' else 1
        fare = st.number_input("Fare ($)", min_value=0.0, max_value=600.0, value=32.0)
        embarked = st.selectbox("Port of Embarkation", options=["C", "Q", "S"])
        em={'C':0,'Q':1,'S':2}
        embarkedd=em[embarked]
    with col2:
        age = st.slider("Age", min_value=0, max_value=100, value=25)
        sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.slider("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
        butt=st.button("Predict")

# User Inputs List
inputs=[[ age, pclass, sexx, sibsp, parch, fare, embarkedd]]

# Prediction Display
if butt:
    with st.expander("**Prediction**",expanded=True):
        with st.spinner("Predicting"):
            t.sleep(2)
        st.subheader(f"Titanic Surviver Found By {classifier} Classifier")
        tab1,tab2=st.tabs(["Survival","Chart"])

        with tab1:
            if classifier=='RandomForestClassifier':
                model=model1
            elif classifier=='KNN':
                model=model2
            else:
                model=model3
            
            pred=model.predict(inputs)
            y_pred_all = model.predict(x_test)  

            write =st.success("🎉 This passenger would have **Survived!**") if pred[0]==1 else st.error("💀 This passenger would **Not** have survived.")
            proba = model.predict_proba(inputs)[0]
            st.write(f"🧠 Survival Probability: {proba[1]*100:.2f}%")
            report = classification_report(y_test, y_pred_all, output_dict=True)
            st.metric("Accuracy", f"{report['accuracy']*100:.2f}%")
            st.metric("Precision", f"{report['1']['precision']*100:.2f}%")
            st.metric("Recall", f"{report['1']['recall']*100:.2f}%")

        # Charts
        with tab2:
            if compare:
                models = [model1, model2, model3]
                names = ['Random Forest', 'Logistic Regression', 'KNN']
                scores = [m.score(x_test, y_test) for m in models]
                df_scores = pd.DataFrame({'Model': names, 'Accuracy': scores})
                st.bar_chart(df_scores.set_index('Model'))
            if classifier == 'RandomForestClassifier':
                    importance = model.feature_importances_
                    st.subheader("📊 Feature Importance")
                    imp_df = pd.DataFrame({'Feature': x.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
                    st.bar_chart(imp_df.set_index('Feature'))
            elif classifier == 'LogisticRegression':
                coef = model.coef_[0]
                imp_df = pd.DataFrame({'Feature': x.columns, 'Coefficient': coef}).sort_values(by='Coefficient', key=np.abs, ascending=False)
                st.subheader("📊 Feature Importance")
                fig, ax = plt.subplots()
                sns.barplot(x='Coefficient', y='Feature', data=imp_df, palette='coolwarm', ax=ax)
                st.pyplot(fig)
            elif classifier=='KNN':
                proba_df = pd.DataFrame({
                'Survival': ['Not Survived (0)', 'Survived (1)'],
                'Probability': proba
                })
                fig,ax=plt.subplots()
                sns.barplot(x='Survival',y='Probability',data=proba_df,palette='Blues',ax=ax)
                ax.set_title("Probability of Survival vs Non-Survival")
                st.pyplot(fig)

# Displaying Titanic Dataset        
if raw:
    with st.expander("See Titanic Raw Data",expanded=True):
        st.dataframe(data.head(50))

col1,col2,cl3=st.columns([1,1,1])
with col2:
    if st.button("Restart"):
        st.session_state.clear()
        st.rerun()    # Restart 
