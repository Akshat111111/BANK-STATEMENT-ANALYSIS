import streamlit as st
import pdfplumber
import re
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def process_text_to_df(text):
    transactions = []
    transaction_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
    previous_balance = None
    
    for line in text.split('\n'):
        match = transaction_pattern.search(line)
        if match:
            date_str, description, amount_str, balance_str = match.groups()
            amount = float(amount_str.replace(',', '').replace('R', '').replace(' ', ''))
            balance = float(balance_str.replace(',', '').replace('R', '').replace(' ', ''))
            
            if previous_balance is not None:
                if balance < previous_balance:
                    transaction_type = 'Debit'
                    debit_amount = abs(amount)
                    credit_amount = 0
                else:
                    transaction_type = 'Credit'
                    credit_amount = amount
                    debit_amount = 0
            else:
                transaction_type = 'Debit' if amount < 0 else 'Credit'
                debit_amount = abs(amount) if amount < 0 else 0
                credit_amount = amount if amount > 0 else 0
            
            transactions.append([date_str, description.strip(), amount, balance, transaction_type, debit_amount, credit_amount])
            previous_balance = balance
    
    return pd.DataFrame(transactions, columns=['Date', 'Description', 'Amount', 'Balance', 'Type', 'Debit Amount', 'Credit Amount'])

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
    elif 'electricity' in description_lower:
        return 'Electricity Charges'
    elif 'interest' in description_lower:
        return 'Interest and Fees'
    elif 'unsuccessful' in description_lower:
        return 'Unsuccessful Transactions'
    elif 'realtime credit' in description_lower:
        return 'Real-time Credits'
    else:
        return 'Others'

def compute_metrics(df):
    avg_daily_expense = df['Amount'].mean()
    total_expense = df['Amount'].sum()
    max_expense = df['Amount'].max()
    min_expense = df['Amount'].min()
    num_transactions = len(df)
    num_debits = df[df['Debit Amount'] > 0].shape[0]
    num_credits = df[df['Credit Amount'] > 0].shape[0]
    avg_balance = df['Balance'].mean()
    closing_balance = df['Balance'].iloc[-1]
    
    return avg_daily_expense, total_expense, max_expense, min_expense, num_transactions, num_debits, num_credits, avg_balance, closing_balance

def train_random_forest_classifier(df):
    training_df = pd.read_csv('nedbank.csv')
    X_train = training_df[['Closing Balance', 'Total Credit', 'Average Balance', 'Number of Transactions', 'Number of Debits', 'Number of Credits']]
    y_train = training_df['Eligibility']

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    avg_daily_expense, total_expense, max_expense, min_expense, num_transactions, num_debits, num_credits, avg_balance, closing_balance = compute_metrics(df)
    loan_eligibility_prediction = classifier.predict([[closing_balance, total_expense, avg_balance, num_transactions, num_debits, num_credits]])[0]
    
    return loan_eligibility_prediction

def main():
    st.title('Upload Your Bank Statement')
    
    st.write(
        """
        <style>
        .success-message {
            font-size: 1.25rem;
            color: green;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="success-message">Congratulations! You\'ve successfully passed the initial eligibility check. Please submit your bank statement so we can complete the final eligibility review.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader('Please upload your bank statement', type='pdf')
    
    if uploaded_file is not None:
        st.write(f'Filename: {uploaded_file.name}')

        text = parse_pdf(uploaded_file)
        df = process_text_to_df(text)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        if not df.empty:
            df['Category'] = df['Description'].apply(categorize_expense)
            loan_eligibility = train_random_forest_classifier(df)

            if loan_eligibility:
                st.markdown('<p style="color:green;">The user is eligible for a loan.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:red;">The user is not eligible for a loan.</p>', unsafe_allow_html=True)

            st.subheader('Parsed Data')
            st.write(df)

            avg_daily_expense, total_expense, max_expense, min_expense, num_transactions, num_debits, num_credits, avg_balance, closing_balance = compute_metrics(df)

            st.subheader('Key Metrics')
            st.write(f'Average Daily Expense: R{avg_daily_expense:.2f}')
            st.write(f'Total Expense: R{total_expense:.2f}')
            st.write(f'Maximum Expense: R{max_expense:.2f}')
            st.write(f'Minimum Expense: R{min_expense:.2f}')
            st.write(f'Number of Transactions: {num_transactions}')
            st.write(f'Number of Debits: {num_debits}')
            st.write(f'Number of Credits: {num_credits}')
            st.write(f'Average Balance: R{avg_balance:.2f}')
            st.write(f'Closing Balance: R{closing_balance:.2f}')

            st.subheader('Expense Overview')

            fig_bar = px.bar(df, x='Date', y='Amount', color='Category', title='Total Expenses per Date')
            st.plotly_chart(fig_bar)

            fig_pie_category = px.pie(df, values='Amount', names='Category', title='Expense Distribution by Category')
            st.plotly_chart(fig_pie_category)

            fig_pie_description = px.pie(df, values='Amount', names='Description', title='Expense Distribution by Description')
            st.plotly_chart(fig_pie_description)

            fig_line = px.line(df, x='Date', y='Amount', title='Daily Expense Trend')
            st.plotly_chart(fig_line)
        else:
            st.write("No transactions found in the uploaded statement.")
    else:
        st.write("Please upload a PDF file.")

if __name__ == '__main__':
    main()
