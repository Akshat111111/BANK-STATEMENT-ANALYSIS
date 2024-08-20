import streamlit as st
import pdfplumber
import re
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Bank Statement Analysis', page_icon=':moneybag:')

st.markdown("""
    <h1 style="text-align: center;">Upload Your Bank Statement</h1>
    <h3 style='text-align: center; color: green;'>Congratulations! You've successfully passed the initial eligibility check. Please submit your bank statement so we can complete the final eligibility review.</h3>
    """, unsafe_allow_html=True)

st.write('<p style="text-align: center;">Please upload your bank statement</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop file here", type="pdf", label_visibility="collapsed")

def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def process_text_to_df(text):
    transactions = []
    transaction_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})\s+(.+?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
    
    for line in text.split('\n'):
        match = transaction_pattern.search(line)
        if match:
            date_str, description, amount_str, balance_str = match.groups()
            amount = float(amount_str.replace(',', '').replace('R', '').replace(' ', ''))
            balance = float(balance_str.replace(',', '').replace('R', '').replace(' ', ''))
            transactions.append([date_str, description.strip(), amount, balance])
    
    return pd.DataFrame(transactions, columns=['Date', 'Description', 'Amount', 'Balance'])

def categorize_expense(description):
    description_lower = description.lower()
    if 'cashsend mobile' in description_lower:
        return 'POS Purchases'
    elif 'immediate trf' in description_lower or 'digital payment' in description_lower:
        return 'Payments'
    elif 'acb credit' in description_lower or 'immediate trf cr' in description_lower:
        return 'Credits'
    elif 'fees' in description_lower or 'charge' in description_lower:
        return 'Bank Charges'
    elif 'atm' in description_lower or 'cash deposit' in description_lower:
        return 'Cash Deposits/Withdrawals'
    elif 'airtime' in description_lower:
        return 'Cellular Expenses'
    elif 'interest' in description_lower:
        return 'Interest and Fees'
    elif 'unsuccessful' in description_lower:
        return 'Unsuccessful Transactions'
    elif 'realtime credit' in description_lower:
        return 'Real-time Credits'
    else:
        return 'Others'

def prepare_features(df):
    total_credits = df[df['Amount'] > 0]['Amount'].sum()
    total_debits = abs(df[df['Amount'] < 0]['Amount'].sum())
    num_transactions = len(df)
    return total_credits, total_debits, num_transactions

def compute_metrics(df):
    avg_daily_expense = df['Amount'].mean()
    total_expense = df['Amount'].sum()
    max_expense = df['Amount'].max()
    min_expense = df['Amount'].min()
    num_transactions = len(df)
    return avg_daily_expense, total_expense, max_expense, min_expense, num_transactions

def build_random_forest_model():
    df_data = pd.read_csv('absa.csv')
    X = df_data[['total_credits', 'total_debits', 'num_transactions', 'avg_transaction_amount', 'transaction_variability', 'balance_trend']]
    y = df_data['Eligibility (y)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def plot_expense_distribution(df):
    categories = {
        'Credits': 0,
        'Payments': 0,
        'Interest and Fees': 0,
        'Unsuccessful Transactions': 0,
        'Real-time Credits': 0,
        'Cellular Expenses': 0,
        'POS Purchases': 0,
        'Bank Charges': 0
    }
    
    for category, amount in df.groupby('Category')['Amount'].sum().items():
        if category in categories:
            categories[category] = amount
    
    categories_df = pd.DataFrame(list(categories.items()), columns=['Category', 'Amount'])
    
    fig = px.pie(categories_df, values='Amount', names='Category', title='Expense Distribution by Category')
    st.plotly_chart(fig)

if uploaded_file is not None:
    try:
        text = parse_pdf(uploaded_file)
        df = process_text_to_df(text)
        df['Date'] = pd.to_datetime(df['Date'])

        if not df.empty:
            df['Category'] = df['Description'].apply(categorize_expense)
            total_credits, total_debits, num_transactions = prepare_features(df)
            model = build_random_forest_model()

            features = pd.DataFrame({'total_credits': [total_credits],
                                     'total_debits': [total_debits],
                                     'num_transactions': [num_transactions],
                                     'avg_transaction_amount': [df['Amount'].mean()],
                                     'transaction_variability': [df['Amount'].std()],
                                     'balance_trend': [df['Balance'].iloc[-1] - df['Balance'].iloc[0]]})
            
            rf_prediction = model.predict(features)[0]
            eligible = total_credits > abs(total_debits) and total_credits > 1.25 * abs(total_debits) and rf_prediction == 1

            if eligible:
                result = 'Eligible for Loan'
                color = 'green'
            else:
                result = 'Not Eligible for Loan'
                color = 'red'

            st.markdown(f'<p style="color:{color};font-size:24px;">{result}</p>', unsafe_allow_html=True)

            st.subheader('Extracted Transactions')
            st.dataframe(df, use_container_width=True)

            avg_daily_expense, total_expense, max_expense, min_expense, num_transactions = compute_metrics(df)

            st.subheader('Key Metrics')
            st.write(f'Average Daily Expense: R{avg_daily_expense:.2f}')
            st.write(f'Total Expense: R{total_expense:.2f}')
            st.write(f'Maximum Expense: R{max_expense:.2f}')
            st.write(f'Minimum Expense: R{min_expense:.2f}')
            st.write(f'Number of Transactions: {num_transactions}')

            st.subheader('Expense Overview')
            fig_bar = px.bar(df, x='Date', y='Amount', color='Category', title='Total Expenses per Date')
            st.plotly_chart(fig_bar)

            st.subheader('Expense Distribution by Category')
            plot_expense_distribution(df)

            fig_pie_description = px.pie(df, values='Amount', names='Description', title='Expense Distribution by Description')
            st.plotly_chart(fig_pie_description)

            fig_line = px.line(df, x='Date', y='Amount', title='Daily Expense Trend')
            st.plotly_chart(fig_line)
        else:
            st.write("No transactions found in the uploaded statement.")
    except Exception as e:
        st.error(f"Error: {e}")

hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
    content:'Made with ❤️ by Akshat'; 
    visibility: visible;
    display: block;
    position: relative;
    padding: 15px;
    top: 2px;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
