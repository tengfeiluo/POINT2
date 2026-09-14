import openai
import random
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error
from openai import OpenAI

PROPERTY_UNITS = {
    'Tg': ('Glass transition temperature', '°C'),
    'Tm': ('Melting temperature', '°C'),
    'density': ('Density', 'g/cm³'),
    'TC': ('Thermal conductivity', 'W/(m·K)'),
    'Bulk_modulus_GPa': ('Bulk modulus', 'GPa'),
    'FFV': ('Fractional free volume', 'dimensionless'),
    'O2_msa_log10': ('Oxygen gas permeability', 'log₁₀(Barrer)'),
    'N2_msa_log10': ('Nitrogen gas permeability', 'log₁₀(Barrer)'),
    'H2_msa_log10': ('Hydrogen gas permeability', 'log₁₀(Barrer)'),
    'CO2_msa_log10': ('Carbon dioxide gas permeability', 'log₁₀(Barrer)'),
    'CH4_msa_log10': ('Methane gas permeability', 'log₁₀(Barrer)')
}

def generate_llm_prompt(smiles, target_property, train_data=None, few_shot=False, icl_k=5):
    """
    Generate a structured prompt for LLM prediction based on zero-shot or few-shot learning.
    """
    formal_name, unit = PROPERTY_UNITS.get(target_property, (target_property, "unknown units"))
    instruction = (
        f"You are an expert in polymer property prediction. Given the SMILES representation of a polymer, "
        f"predict the {formal_name} ({target_property}) value in {unit} with high accuracy. "
        f"Output ONLY the predicted numerical value, and nothing else. Do NOT include explanations, units, or additional text. "
        f"Note: '*' denotes the polymerization points in the SMILES of the repeat unit of polymers."
    )
    
    if few_shot and train_data is not None:
        examples = train_data.sample(n=icl_k)  # Randomly select examples
        example_text = "\n".join([f"SMILES: {row['SMILES']}, {formal_name} ({target_property}): {row[target_property]}" for _, row in examples.iterrows()])
        prompt = (
            f"{instruction}\n\nHere are some example polymers with their properties:\n"
            f"{example_text}\n\nNow, predict the {formal_name} ({target_property}) value for the following SMILES:\n"
            f"SMILES: {smiles}"
        )
    else:
        prompt = f"{instruction}\n\nPredict the {formal_name} ({target_property}) value for the following SMILES:\nSMILES: {smiles}"
    
    return prompt

def query_gpt4o_mini(prompt, log_file):
    """
    Query GPT-4o-mini for property predictions.
    """
    with open(log_file, 'a') as log:
        log.write(f'=== Prompt ===\n{prompt}\n')
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("GPT-4o-mini loaded successfully!")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an AI model for predicting polymer properties."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    result = response.choices[0].message.content.strip()
    
    with open(log_file, 'a') as log:
        log.write(f'=== Response ===\n{result}\n')
    
    return result

def llm_predict_property(smiles_train, y_train, smiles_test, y_test, target_property, results_dir, few_shot=False, icl_k=5):
    """
    Predict polymer properties using GPT-4o-mini under zero-shot or few-shot settings.
    """
    train_df = pd.DataFrame({'SMILES': smiles_train, target_property: y_train})
    predictions = []
    
    for smiles in smiles_test:
        prompt = generate_llm_prompt(smiles, target_property, train_data=train_df if few_shot else None, few_shot=few_shot, icl_k=icl_k)
        response = query_gpt4o_mini(prompt, os.path.join(results_dir, 'llm_queries.log'))
        
        try:
            pred_value = float(response)  # Extract numerical part
            predictions.append(pred_value)
        except ValueError:
            predictions.append(np.nan)  # Handle missing or incorrect outputs
    
    predictions = np.array(predictions)
    
    # Handle missing predictions by filling with mean prediction
    if np.isnan(predictions).any():
        mean_pred = np.nanmean(predictions)
        predictions = np.where(np.isnan(predictions), mean_pred, predictions)
    
    # Save predictions
    method = "few_shot" if few_shot else "zero_shot"
    np.save(os.path.join(results_dir, f'y_pred_llm_{method}.npy'), predictions)
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"LLM ({method}) RMSE for {target_property}: {rmse:.4f}")
    
    return predictions