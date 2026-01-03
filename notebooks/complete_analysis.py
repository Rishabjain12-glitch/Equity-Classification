"""
Equity Classification Using Financial Ratios
Complete Machine Learning Pipeline

Project: WiDS 5.0 - Equity Classification
Date: January 2026
Objective: Predict investment quality using fundamental financial ratios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

print('All libraries imported successfully!')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
np.random.seed(42)

def generate_financial_data(n_companies=50, n_years=5):
    """Generate synthetic financial statement data for 50 companies across 5 years."""
    data = []
    companies = [f'Company_{i:03d}' for i in range(1, n_companies + 1)]
    sectors = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 'Industrial']
    base_year = 2019
    
    for company_idx, company in enumerate(companies):
        sector = np.random.choice(sectors)
        growth_rate = np.random.uniform(-0.05, 0.15)
        volatility = np.random.uniform(0.05, 0.25)
        base_size = np.random.uniform(100, 10000)
        investment_quality = np.random.choice(['Good', 'Bad'], p=[0.4, 0.6])
        
        for year in range(n_years):
            year_value = base_year + year
            growth_factor = 1 + growth_rate + np.random.normal(0, volatility) * year
            
            revenue = base_size * growth_factor * np.random.uniform(0.9, 1.1)
            cogs = revenue * np.random.uniform(0.60, 0.80)
            gross_profit = revenue - cogs
            operating_expenses = revenue * np.random.uniform(0.15, 0.30)
            ebit = gross_profit - operating_expenses
            interest_expense = abs(np.random.normal(revenue * 0.02, revenue * 0.01))
            ebt = ebit - interest_expense
            tax = max(0, ebt * np.random.uniform(0.20, 0.30))
            net_income = ebt - tax
            
            current_assets = revenue * np.random.uniform(0.3, 0.6)
            cash = current_assets * np.random.uniform(0.2, 0.5)
            accounts_receivable = current_assets * np.random.uniform(0.2, 0.4)
            inventory = current_assets * np.random.uniform(0.1, 0.3)
            fixed_assets = revenue * np.random.uniform(0.5, 1.5)
            total_assets = current_assets + fixed_assets
            
            current_liabilities = revenue * np.random.uniform(0.2, 0.4)
            accounts_payable = current_liabilities * np.random.uniform(0.3, 0.6)
            short_term_debt = current_liabilities * np.random.uniform(0.2, 0.5)
            long_term_debt = revenue * np.random.uniform(0.3, 0.8)
            total_liabilities = current_liabilities + long_term_debt
            shareholders_equity = total_assets - total_liabilities
            
            operating_cash_flow = net_income * np.random.uniform(0.8, 1.4)
            capex = fixed_assets * np.random.uniform(0.05, 0.15)
            free_cash_flow = operating_cash_flow - capex
            
            shares_outstanding = np.random.uniform(50, 500)
            eps = net_income / shares_outstanding if shares_outstanding > 0 else 0
            
            if investment_quality == 'Good':
                pe_ratio = np.random.uniform(15, 30)
            else:
                pe_ratio = np.random.uniform(5, 15)
            
            stock_price = eps * pe_ratio if eps > 0 else np.random.uniform(5, 20)
            market_cap = stock_price * shares_outstanding
            dividend_per_share = max(0, eps * np.random.uniform(0, 0.5) if eps > 0 else 0)
            
            record = {
                'Company': company, 'Sector': sector, 'Year': year_value, 'Quarter': 'Q4',
                'Revenue': revenue, 'COGS': cogs, 'GrossProfit': gross_profit,
                'OperatingExpenses': operating_expenses, 'EBIT': ebit,
                'InterestExpense': interest_expense, 'EBT': ebt, 'Tax': tax,
                'NetIncome': net_income, 'CurrentAssets': current_assets, 'Cash': cash,
                'AccountsReceivable': accounts_receivable, 'Inventory': inventory,
                'FixedAssets': fixed_assets, 'TotalAssets': total_assets,
                'CurrentLiabilities': current_liabilities, 'AccountsPayable': accounts_payable,
                'ShortTermDebt': short_term_debt, 'LongTermDebt': long_term_debt,
                'TotalLiabilities': total_liabilities, 'ShareholdersEquity': shareholders_equity,
                'OperatingCashFlow': operating_cash_flow, 'CapEx': capex,
                'FreeCashFlow': free_cash_flow, 'SharesOutstanding': shares_outstanding,
                'StockPrice': stock_price, 'MarketCap': market_cap, 'EPS': eps,
                'DividendPerShare': dividend_per_share, 'InvestmentQuality': investment_quality
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_cols = ['DividendPerShare', 'FreeCashFlow', 'CapEx']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df

def compute_financial_ratios(df):
    """Compute 19 financial ratios from raw data."""
    ratios = {}
    
    # Profitability Ratios
    ratios['GrossProfitMargin'] = (df['GrossProfit'] / df['Revenue']) * 100
    ratios['OperatingProfitMargin'] = (df['EBIT'] / df['Revenue']) * 100
    ratios['NetProfitMargin'] = (df['NetIncome'] / df['Revenue']) * 100
    ratios['ROA'] = (df['NetIncome'] / df['TotalAssets']) * 100
    ratios['ROE'] = (df['NetIncome'] / df['ShareholdersEquity']) * 100
    ratios['OCFMargin'] = (df['OperatingCashFlow'] / df['Revenue']) * 100
    
    # Liquidity Ratios
    ratios['CurrentRatio'] = df['CurrentAssets'] / df['CurrentLiabilities']
    ratios['QuickRatio'] = (df['CurrentAssets'] - df['Inventory']) / df['CurrentLiabilities']
    ratios['CashRatio'] = df['Cash'] / df['CurrentLiabilities']
    
    # Leverage Ratios
    ratios['DebttoEquity'] = df['TotalLiabilities'] / df['ShareholdersEquity']
    ratios['DebttoAssets'] = df['TotalLiabilities'] / df['TotalAssets']
    ratios['EquityRatio'] = df['ShareholdersEquity'] / df['TotalAssets']
    ratios['InterestCoverage'] = df['EBIT'] / df['InterestExpense']
    
    # Efficiency Ratios
    ratios['AssetTurnover'] = df['Revenue'] / df['TotalAssets']
    ratios['InventoryTurnover'] = df['COGS'] / df['Inventory']
    ratios['ReceivablesTurnover'] = df['Revenue'] / df['AccountsReceivable']
    
    # Valuation Ratios
    ratios['PERatio'] = df['StockPrice'] / df['EPS']
    ratios['PBRatio'] = df['MarketCap'] / df['ShareholdersEquity']
    ratios['DividendYield'] = (df['DividendPerShare'] / df['StockPrice']) * 100
    
    return ratios

if __name__ == '__main__':
    print('Generating financial data...')
    df_raw = generate_financial_data(n_companies=50, n_years=5)
    print(f'Generated {len(df_raw)} records for {df_raw["Company"].nunique()} companies')
    print(f'Shape: {df_raw.shape}')
    print(f'Years: {df_raw["Year"].min()} - {df_raw["Year"].max()}')
    
    # Handle missing values
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='median')
    df_raw[numeric_cols] = imputer.fit_transform(df_raw[numeric_cols])
    
    # Compute ratios
    print('\nComputing financial ratios...')
    ratios = compute_financial_ratios(df_raw)
    ratios_df = pd.DataFrame(ratios)
    df_with_ratios = pd.concat([df_raw, ratios_df], axis=1)
    
    # Replace infinite values
    inf_count = np.isinf(df_with_ratios.select_dtypes(include=[np.number])).sum().sum()
    df_with_ratios = df_with_ratios.replace([np.inf, -np.inf], np.nan)
    
    # Cap extreme values
    ratio_columns = list(ratios.keys())
    for col in ratio_columns:
        if df_with_ratios[col].dtype in ['float64', 'int64']:
            p1 = df_with_ratios[col].quantile(0.01)
            p99 = df_with_ratios[col].quantile(0.99)
            df_with_ratios[col] = df_with_ratios[col].clip(lower=p1, upper=p99)
    
    # Final imputation
    imputer_ratios = SimpleImputer(strategy='median')
    df_with_ratios[ratio_columns] = imputer_ratios.fit_transform(df_with_ratios[ratio_columns])
    
    print(f'Total ratios computed: {len(ratios)}')
    print(f'Dataset shape: {df_with_ratios.shape}')
    
    # Save preprocessed data
    df_with_ratios.to_csv('data/processed_financial_data.csv', index=False)
    print('\nProcessed data saved to data/processed_financial_data.csv')
    print('Analysis complete!')
