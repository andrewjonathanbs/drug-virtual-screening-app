#!/usr/bin/env python
# coding: utf-8


# In[ ]:
import streamlit as st
import pandas as pd
import numpy as np
import pycaret
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from pycaret.classification import *
import base64


# In[ ]:


#CSV Data Input
st.title("SimpleScreen: Machine Learning Model Creator for Virtual Screening")
st.subheader("Created by Andrew Jonathan Brahms Simangunsong.")
st.write("In this site, you can create machine learning models to virtual screen drug candidates without any code!")

st.subheader("How SimpleScreen works:")
st.write("SimpleScreen will perform feature extraction on the molecule Smiles then automatically create classifier machine learning models")
st.write("based on the standard value (IC50 or EC50). Then it will choose the best model based on Accuracy, AUC score, Recall, Precision, F1,")
st.write("Kappa score, MCC, and Time Taken. Then, it will tune the model and predict other dataset with the model.")

st.subheader("How to use SimpleScreen:")
st.write("First, upload your data in CSV format.")
st.write("Make sure the data contains Smiles in a column named Smiles and the Standard Value of the activity type in a column named Standard Value")
st.write("After that, we shall create machine learning models based on your data and choose the best model for you to use.")
st.write("Then, you can upload the data you want to predict.")
st.write("Make sure the data contains Smiles in a column named Smiles.")


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

uploaded_pred_file = st.file_uploader("Upload a CSV file to be predicted", type=["csv"])
if uploaded_pred_file is not None:
    df_pred = pd.read_csv(uploaded_pred_file)


# In[ ]:


# Data Processing
def data_processing(df):
    df2 = df[df['Standard Value'].notna()]  # Filter rows where 'Standard Value' is not NaN
    df2 = df2[df2['Smiles'].notna()]  # Filter rows where 'Smiles' is not NaN
    df2 = df2.drop_duplicates(subset=['Smiles'])  # Remove duplicate rows based on 'Smiles'
    return df2

def data_processing_nosv(df):
    df2 = df[df['Smiles'].notna()]  # Filter rows where 'Smiles' is not NaN
    df21 = df2.drop_duplicates(subset=['Smiles'])  # Remove duplicate rows based on 'Smiles'
    return df21


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
    df2 = df[df['Smiles'].notna()]  # Filter rows where 'Smiles' is not NaN
    df2 = df2.drop_duplicates(subset=['Smiles'])  # Remove duplicate rows based on 'Smiles'
    df2['Smiles'] = df2['Smiles'].astype(str)
    return df2


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

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

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


if st.button("Make ML Models"):
        progress_bar = st.progress(0)
        st.write("Preprocess the data...")
        data_processing(df)
        df3 = log_the_smiles(df)
        df4 = df3[['Smiles', 'class']]
        df5 = data_processing_1(df4)
    
        progress_bar.progress(12)
    
        st.write("Extracting 200 smiles features...")
        Mol_descriptors,desc_names = RDkit_descriptors(df5['Smiles'])
        df_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
        df5_reset = df5.reset_index(drop=True)
        df_with_200_descriptors_reset = df_with_200_descriptors.reset_index(drop=True)
        result_df_1 = pd.concat([df5_reset, df_with_200_descriptors_reset], axis=1)

        progress_bar.progress(25)
    
        st.write("Extracting 2000 smiles features")
        Morgan_fpts = morgan_fpts(df5['Smiles'])
        Morgan_fingerprints = pd.DataFrame(Morgan_fpts,columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])
        result_df_1_reset = result_df_1.reset_index(drop=True)
        Morgan_fingerprints_reset = Morgan_fingerprints.reset_index(drop=True)
        result_df_2 = pd.concat([result_df_1_reset,Morgan_fingerprints_reset], axis=1)
        result_df_2 = result_df_2.dropna(subset=['Smiles','class'])
        result_df_3 = result_df_2.drop('Smiles', axis=1)
        experiment = setup(result_df_3, target='class')

        progress_bar.progress(38)
    
        st.write("Comparing ML models...")
        best_model = compare_models()

        progress_bar.progress(51)

        st.write("Preparing the best model...")
        tuned_model = create_model(best_model)
        final_model = finalize_model(best_model)

        progress_bar.progress(63)

        st.write("Predictions are in progress...")
        st.write("Preprocessing the data...")
        valid_mask = df_pred['Smiles'].apply(is_valid_smiles)
        valid_df = df_pred[valid_mask]
        dfa = data_processing_nosv(valid_df)
        dfb = data_processing_1(dfa)
        st.write("Extracting molecular features from Smiles")

        progress_bar.progress(76)
    
        Mol_descriptorsc,desc_namesc = RDkit_descriptors(dfb['Smiles'])
        dfb_with_200_descriptors = pd.DataFrame(Mol_descriptorsc,columns=desc_namesc)
        dfb_reset = dfb.reset_index(drop=True)
        dfb_with_200_descriptors_reset = dfb_with_200_descriptors.reset_index(drop=True)
        result_dfb_1 = pd.concat([dfb_reset, dfb_with_200_descriptors_reset], axis=1)
        Morgan_fptsc = morgan_fpts(dfb['Smiles'])
        Morgan_fingerprintsc = pd.DataFrame(Morgan_fptsc,columns=['Col_{}'.format(i) for i in range(Morgan_fptsc.shape[1])])
        result_dfb_1_reset = result_dfb_1.reset_index(drop=True)
        Morgan_fingerprintsc_reset = Morgan_fingerprintsc.reset_index(drop=True)

        progress_bar.progress(88)

        st.write("Applying ML model to the data...")
        result_dfb_2 = pd.concat([result_dfb_1_reset,Morgan_fingerprintsc_reset], axis=1)
        result_dfb_2 = result_dfb_2.dropna(subset=['Smiles'])
        predictions = predict_model(best_model, data= result_dfb_2)
        dfp = pd.DataFrame(predictions)
        valid_df['predicted_class'] = dfp['prediction_label']
        valid_df['prediction_score'] = dfp['prediction_score']

        progress_bar.progress(101)
    
        st.write("Prediction complete!")
        def create_download_link(df, filename="predictions.csv"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
            return href
        # Create and display the download link
        st.markdown(create_download_link(valid_df), unsafe_allow_html=True)
