#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from pycaret.classification import *


# In[ ]:


#CSV Data Input
def main():
    st.title("Machine Learning Creator for Drug Discovery")
    st.subheader("In this site, you can create machine learning models to predict drug activity without any code!")
    st.write("First, upload your data in CSV format.")
    st.write("""Make sure the data contains Smiles in a column named "Smiles" and the Standard Value of the activity type in a column named "Standard Value""")
    st.write("After that, the app shall create machine learning models based on your data and choose the best model based on your data.")
    st.write("Then, you can upload the data you want to predict.")
    st.write("""Make sure the data contains Smiles in a columen named "Smiles""")

def train_uploader():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.write(df)


# In[ ]:


# Data Processing
def data_processing(df):
    df2 = df[df['Standard Value'].notna()]  # Filter rows where 'Standard Value' is not NaN
    df2 = df2[df2['Smiles'].notna()]  # Filter rows where 'Smiles' is not NaN
    df2 = df2.drop_duplicates(subset=['Smiles'])  # Remove duplicate rows based on 'Smiles'
    return df2


# In[ ]:


def log_the_smiles(df):
    if df['Standard Value'].mean() in range(0, 10):
        pass
    else:
        bioactivity_threshold = []
        for i in df['Standard Value']:
            if float(i) >= 10000:
                bioactivity_threshold.append("inactive")
            elif float(i) <= 1000:
                bioactivity_threshold.append("active")
            else:
                bioactivity_threshold.append("intermediate")

        bioactivity_class = pd.Series(bioactivity_threshold, name='class')
        df3 = pd.concat([df, bioactivity_class], axis=1)
        df3 = df3[df3['class'] != 'intermediate']  # Use correct syntax to drop rows
        return df3


# In[ ]:


# Data Processing
def data_processing_1(df):
    df2 = df[df['class'].notna()]  # Filter rows where 'Standard Value' is not NaN
    df2 = df2[df2['Smiles'].notna()]  # Filter rows where 'Smiles' is not NaN
    df2 = df2.drop_duplicates(subset=['Smiles'])  # Remove duplicate rows based on 'Smiles'
    df4['Smiles'] = df4['Smiles'].astype(str)
    return df2


# In[ ]:


df5 = data_processing_1(df4)


# In[ ]:


#Feature Extraction
def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles] 
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:
        # add hydrogens to molecules
        mol=Chem.AddHs(mol)
        # Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names 


# In[ ]:


# Assuming both DataFrames have consistent columns
# Reset the indices of the DataFrames before concatenation
df5_reset = df5.reset_index(drop=True)
df_with_200_descriptors_reset = df_with_200_descriptors.reset_index(drop=True)

# Concatenate the DataFrames
result_df_1 = pd.concat([df5_reset, df_with_200_descriptors_reset], axis=1)


# In[ ]:


def morgan_fpts(data):
    Morgan_fpts = []
    for i in data:
        mol = Chem.MolFromSmiles(i) 
        fpts =  AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        mfpts = np.array(fpts)
        Morgan_fpts.append(mfpts)  
    return np.array(Morgan_fpts)


# In[ ]:


tuned_model = create_model(best_model)


# In[ ]:


final_model = finalize_model(best_model)


# In[ ]:


if st.button("Make Predictions"):
        st.write("Predictions are in progress...")
        data_processing(df)
        df3 = log_the_smiles(df2)
        df4 = df3[['Smiles', 'class']]
        df5 = data_processing_1(df4)
        st.write("Extracting smiles features...")
        Mol_descriptors,desc_names = RDkit_descriptors(df5['Smiles'])
        
        df5_reset = df5.reset_index(drop=True)
        df_with_200_descriptors_reset = df_with_200_descriptors.reset_index(drop=True)
        result_df_1 = pd.concat([df5_reset, df_with_200_descriptors_reset], axis=1)
        Morgan_fpts = morgan_fpts(df5['Smiles'])
        Morgan_fingerprints = pd.DataFrame(Morgan_fpts,columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])
        result_df_1_reset = result_df_1.reset_index(drop=True)
        Morgan_fingerprints_reset = Morgan_fingerprints.reset_index(drop=True)
        result_df_2 = pd.concat([result_df_1_reset,Morgan_fingerprints_reset], axis=1)
        result_df_2 = result_df_2.dropna(subset=['Smiles','class'])
        result_df_3 = result_df_2.drop('Smiles', axis=1)
        experiment = setup(result_df_3, target='class')
        st.write("Comparing ML models...")
        best_model = compare_models()
        st.write("Preparing the best model...")
        tuned_model = create_model(best_model)
        final_model = finalize_model(best_model)
        st.write("Model Completed! Now, uplade the data you want to predict!")


# In[ ]:


def predict_uploader():
    uploaded_pred_file = st.file_uploader("Upload a CSV file to be predicted", type=["csv"])
    if uploaded_file is not None:
        df_pred = pd.read_csv(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.write(dfz)


# In[ ]:


if st.button("Predict activity"):
        st.write("Predictions are in progress...")
        st.write("Preprocessing the data...")
        dfa = data_processing(dfz)
        dfb = log_the_smiles(dfa)
        dfc = data_processing_1(dfb)
        st.write("Extracting molecular features from Smiles")
        Mol_descriptorsc,desc_namesc = RDkit_descriptors(dfc['Smiles'])
        dfc_with_200_descriptors = pd.DataFrame(Mol_descriptorsc,columns=desc_namesc)
        dfc_reset = dfc.reset_index(drop=True)
        dfc_with_200_descriptors_reset = dfc_with_200_descriptors.reset_index(drop=True)
        result_dfc_1 = pd.concat([dfc_reset, dfc_with_200_descriptors_reset], axis=1)
        Morgan_fptsc = morgan_fpts(dfc['Smiles'])
        Morgan_fingerprintsc = pd.DataFrame(Morgan_fptsc,columns=['Col_{}'.format(i) for i in range(Morgan_fptsc.shape[1])])
        result_dfc_1_reset = result_dfc_1.reset_index(drop=True)
        Morgan_fingerprintsc_reset = Morgan_fingerprintsc.reset_index(drop=True)
        st.write("Applying ML model to the data...")
        result_dfc_2 = pd.concat([result_dfc_1_reset,Morgan_fingerprintsc_reset], axis=1)
        result_dfc_2 = result_dfc_2.dropna(subset=['Smiles','class'])
        result_dfc_3 = result_dfc_2.drop('Smiles', axis=1)
        predictions = predict_model(best_model, data=result_dfc_3)
        dfz['predicted_class'] = predictions['Label']
        st.write("Prediction complete!")


# In[ ]:


def create_download_link(df, filename="predictions.csv"):
    csv = dfz.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

# Display the DataFrame with predicted labels
st.write(result_dfc_3)

# Create and display the download link
st.markdown(create_download_link(result_dfc_3), unsafe_allow_html=True)

