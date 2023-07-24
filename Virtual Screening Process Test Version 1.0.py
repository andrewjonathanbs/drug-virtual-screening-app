#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install pycaret --user')


# In[8]:


get_ipython().system('pip install --upgrade pycaret scikit-learn --user')


# In[1]:


import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from pycaret.classification import *


# In[2]:


df = pd.read_csv("C:\\Users\\ASUS TUFF GAMING\\Downloads\\MCHR1 Data\\Data MCHR1.csv")
df.head()


# In[3]:


# Data Processing
def data_processing(df):
    df2 = df[df['Standard Value'].notna()]  # Filter rows where 'Standard Value' is not NaN
    df2 = df2[df2['Smiles'].notna()]  # Filter rows where 'Smiles' is not NaN
    df2 = df2.drop_duplicates(subset=['Smiles'])  # Remove duplicate rows based on 'Smiles'
    return df2


# In[4]:


df2 = data_processing(df)


# In[5]:


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


# In[6]:


df3 = log_the_smiles(df2)


# In[7]:


df4 = df3[['Smiles', 'class']]


# In[8]:


# Data Processing
def data_processing_1(df):
    df2 = df[df['class'].notna()]  # Filter rows where 'Standard Value' is not NaN
    df2 = df2[df2['Smiles'].notna()]  # Filter rows where 'Smiles' is not NaN
    df2 = df2.drop_duplicates(subset=['Smiles'])  # Remove duplicate rows based on 'Smiles'
    df4['Smiles'] = df4['Smiles'].astype(str)
    return df2


# In[9]:


df5 = data_processing_1(df4)


# In[16]:


df5.to_csv("C:\\Users\\ASUS TUFF GAMING\\Downloads\\MCHR1 Data\\df.csv")
df5


# In[11]:


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

# Function call
Mol_descriptors,desc_names = RDkit_descriptors(df5['Smiles'])


# In[12]:


df_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
df_with_200_descriptors


# In[17]:


# Assuming both DataFrames have consistent columns
# Reset the indices of the DataFrames before concatenation
df5_reset = df5.reset_index(drop=True)
df_with_200_descriptors_reset = df_with_200_descriptors.reset_index(drop=True)

# Concatenate the DataFrames
result_df_1 = pd.concat([df5_reset, df_with_200_descriptors_reset], axis=1)


# In[18]:


result_df_1


# In[19]:


def morgan_fpts(data):
    Morgan_fpts = []
    for i in data:
        mol = Chem.MolFromSmiles(i) 
        fpts =  AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        mfpts = np.array(fpts)
        Morgan_fpts.append(mfpts)  
    return np.array(Morgan_fpts)


# In[20]:


Morgan_fpts = morgan_fpts(df5['Smiles'])


# In[21]:


Morgan_fingerprints = pd.DataFrame(Morgan_fpts,columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])


# In[61]:


result_df_2 = pd.concat([result_df_1,Morgan_fingerprints], ignore_index=True)
print(result_df_2.head())


# In[22]:


# Assuming both DataFrames have consistent columns
# Reset the indices of the DataFrames before concatenation
result_df_1_reset = result_df_1.reset_index(drop=True)
Morgan_fingerprints_reset = Morgan_fingerprints.reset_index(drop=True)

# Concatenate the DataFrames
result_df_2 = pd.concat([result_df_1_reset,Morgan_fingerprints_reset], axis=1)


# In[23]:


result_df_2


# In[24]:


result_df_2 = result_df_2.dropna(subset=['Smiles','class'])


# In[25]:


result_df_3 = result_df_2.drop('Smiles', axis=1)


# In[26]:


experiment = setup(result_df_3, target='class')


# In[27]:


#Compare Models
best_model = compare_models()


# In[28]:


result_df_4 = result_df_1.drop('Smiles', axis=1)


# In[29]:


experiment = setup(result_df_4, target='class')


# In[30]:


best_model = compare_models()


# In[32]:


# Assuming both DataFrames have consistent columns
# Reset the indices of the DataFrames before concatenation
df5_reset = df5.reset_index(drop=True)
Morgan_fingerprints_reset = Morgan_fingerprints.reset_index(drop=True)

# Concatenate the DataFrames
result_df_6 = pd.concat([df5_reset,Morgan_fingerprints_reset], axis=1)


# In[33]:


experiment = setup(result_df_6, target='class')


# In[34]:


best_model = compare_models()


# In[37]:


import plotly.express as px


# In[49]:


fig = px.histogram(result_df_6['class'], nbins=2, labels={'count': 'Frequency', 'x': 'Values'},
                   title='Class Distribution')

# Show the plot
fig.show()


# In[ ]:


#Oh well, no wonder.

