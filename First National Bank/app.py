import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import tempfile

def extract_bank_statement_data(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    closing_balance = re.search(r"Closing Balance\s+([\d,]+\.?\d*)Dr", text)
    credit_transactions = re.search(r"No\. Credit Transactions\s+(\d+)", text)
    total_credit = re.search(r"No\. Credit Transactions\s+\d+\s+([\d,]+\.?\d*)Cr", text)
    name_match = re.search(r"(MR|MRS)\s+([A-Z\s]+)", text)
    if closing_balance and credit_transactions and total_credit:
        closing_balance = float(closing_balance.group(1).replace(',', ''))
        credit_transactions = int(credit_transactions.group(1))
        total_credit = float(total_credit.group(1).replace(',', ''))
        account_holder_name = name_match.group(0) if name_match else "Name not found"
        return closing_balance, credit_transactions, total_credit, account_holder_name
    return None, None, None, None

def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def process_text_to_df(text):
    lines = text.split('\n')
    data = []
    for line in lines:
        try:
            match = re.match(r'(\d{2} \w{3}) (.+?) (\d+,\d{2}|\d+\.\d{2})(Cr|Dr)? (\d+,\d{2}|\d+\.\d{2})(.*)', line)
            if match:
                date = match.group(1).strip()
                description = match.group(2).strip()
                amount_str = match.group(3).replace(',', '').strip()
                amount = float(amount_str)
                if match.group(4) == 'Cr':
                    amount = -amount
                balance_str = match.group(5).replace(',', '').strip()
                balance = float(balance_str)
                data.append({
                    'Date': date,
                    'Description': description,
                    'Amount': amount,
                    'Balance': balance,
                    'Category': categorize_expense(description)
                })
        except Exception as e:
            st.error(f"Error processing line: {line}\n{str(e)}")
            continue
    return pd.DataFrame(data)

def categorize_expense(description):
    description_lower = description.lower()
    if 'pos purchase pnp crp winkelsprui' in description_lower:
        return 'POS Purchases'
    elif 'fnb app rtc pmt to' in description_lower or 'internet pmt to payfast*hollywoodbet' in description_lower:
        return 'Payments'
    elif 'magtape credit inlimeloanlimeloan' in description_lower or 'fnb app payment from creditworth' in description_lower:
        return 'Credits'
    elif 'service fees' in description_lower or 'other fees' in description_lower or 'monthly account fee' in description_lower:
        return 'Bank Charges'
    elif 'atm cash' in description_lower or 'cash deposit fees' in description_lower:
        return 'Cash Deposits/Withdrawals'
    elif 'fnb app prepaid airtime' in description_lower or 'airtime topup airtime' in description_lower or 'device payment' in description_lower:
        return 'Cellular Expenses'
    elif 'item paid no funds' in description_lower or 'hybrid subscription fee' in description_lower:
        return 'Interest and Fees'
    elif 'debit card pos unsuccessful f #fee declined purch tran' in description_lower:
        return 'Unsuccessful Transactions'
    elif 'realtime credit fasta p10001419869' in description_lower:
        return 'Real-time Credits'
    else:
        return 'Others'

def compute_metrics(df):
    avg_daily_expense = df['Amount'].mean()
    total_expense = df['Amount'].sum()
    max_expense = df['Amount'].max()
    min_expense = df['Amount'].min()
    num_transactions = len(df)
    return avg_daily_expense, total_expense, max_expense, min_expense, num_transactions

# Load the training dataset
df_user = pd.read_csv('fnb.csv')
X = df_user[['Closing Balance', 'Credit Transactions', 'Total Credit']]
y = df_user['Eligibility'].apply(lambda x: 1 if x == 'Eligible' else 0)

# Train the Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X, y)

def main():
    st.set_page_config(layout="centered")
    st.title("Upload Your Bank Statement")
    st.markdown(
        "<h3 style='color: green;'>Congratulations! You've successfully passed the initial eligibility check. "
        "Please submit your bank statement so we can complete the final eligibility review.</h3>",
        unsafe_allow_html=True,
    )
    st.write("Please upload your bank statement")
    uploaded_file = st.file_uploader(
        "Drag and drop file here", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        closing_balance, credit_transactions, total_credit, account_holder_name = extract_bank_statement_data(temp_file_path)

        if closing_balance is not None:
            st.write(f"Account Holder: {account_holder_name}")
            X_test = np.array([[closing_balance, credit_transactions, total_credit]])
            eligibility = classifier.predict(X_test)[0]

            if eligibility:
                st.success("The person is eligible for a loan.")
            else:
                st.error("The person is not eligible for a loan.")
        else:
            st.error("Failed to extract data from the bank statement.")

        st.subheader('Uploaded Bank Statement')
        st.write(f'Filename: {uploaded_file.name}')
        text = parse_pdf(uploaded_file)
        df = process_text_to_df(text)

        if not df.empty:
            st.subheader('Parsed Data')
            st.write(df)
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
