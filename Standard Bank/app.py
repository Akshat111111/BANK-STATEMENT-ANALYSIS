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
    transaction_pattern = re.compile(r'(\d{2} \w{3} \d{2})\s+(.+?)\s+(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
    
    for line in text.split('\n'):
        match = transaction_pattern.search(line)
        if match:
            date_str, description, amount_str, balance_str = match.groups()
            amount = float(amount_str.replace(',', '').replace(' ', ''))
            balance = float(balance_str.replace(',', '').replace(' ', ''))
            transactions.append([date_str, description.strip(), amount, balance])
    
    return pd.DataFrame(transactions, columns=['Date', 'Description', 'Amount', 'Balance'])

def categorize_expense(description):
    description_lower = description.lower()
    if 'alaries salary' in description_lower:
        return 'Salary'
    elif 'mtn prepaid' in description_lower or 'fee - pre-paid top up' in description_lower or 'excess interest' in description_lower:
        return 'Mobile Expenses'
    elif re.search(r'\b(?:0000\w+|10134635130)\b', description_lower):
        return 'Cash'
    elif 'fixed monthly fee' in description_lower:
        return 'Monthly Fee'
    elif 'statement costs' in description_lower:
        return 'Statement Cost'
    elif 'cashsend mobile' in description_lower:
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
    return avg_daily_expense, total_expense, max_expense, min_expense, num_transactions

def train_random_forest_classifier():
    data = pd.read_csv('standardbank.csv')
    
    X = data[['Total_Credits', 'Total_Debits', 'Average_Balance', 'Num_Transactions']]
    y = data['Loan_Eligibility']

    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X, y)

    return clf

def predict_loan_eligibility(clf, total_credits, total_debits, avg_balance, num_transactions):
    if total_credits <= 1.25 * abs(total_debits):
        return 'Not Eligible'
    
    features = pd.DataFrame([[total_credits, total_debits, avg_balance, num_transactions]], 
                            columns=['Total_Credits', 'Total_Debits', 'Average_Balance', 'Num_Transactions'])
    prediction = clf.predict(features)
    
    return 'Eligible' if prediction[0] == 1 else 'Not Eligible'

def main():
    st.markdown(
        """
        <style>
        .stFileUploader > label {
            font-size: 1rem;
            font-weight: 400;
            color: #6c757d;
        }
        .stFileUploader div[role="button"] {
            border-radius: 8px;
        }
        .upload-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 50px;
        }
        .upload-section h1 {
            font-size: 2.5rem;
            color: #333;
        }
        .congrats-text {
            color: #28a745;
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown("<h1 style='text-align: center;'>Upload Your Bank Statement</h1>", unsafe_allow_html=True)
    
    st.markdown('<p class="congrats-text">Congratulations! You\'ve successfully passed the initial eligibility check. Please submit your bank statement so we can complete the final eligibility review.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Please upload your bank statement", type="pdf")

    if uploaded_file is not None:
        text = parse_pdf(uploaded_file)
        df = process_text_to_df(text)
        df['Date'] = pd.to_datetime(df['Date'], format='%d %b %y')

        if not df.empty:
            df['Category'] = df['Description'].apply(categorize_expense)
            
            category_dict = df[df['Category'] != 'Others'].groupby('Category')['Amount'].sum().to_dict()
            
            selected_categories = ["Bank Charges", "Statement Cost","Monthly Fee", "Cash Deposits/Withdrawals", "Mobile Expenses", "Salary"]
            filtered_category_dict = {k: v for k, v in category_dict.items() if k in selected_categories}
            
            abs_category_dict = {k: abs(v) for k, v in filtered_category_dict.items()}
            
            total_credits = df[df['Amount'] > 0]['Amount'].sum()
            total_debits = df[df['Amount'] < 0]['Amount'].sum()
            avg_balance = df['Balance'].mean()
            num_transactions = len(df)

            clf = train_random_forest_classifier()

            loan_eligibility = predict_loan_eligibility(clf, total_credits, total_debits, avg_balance, num_transactions)
            
            st.subheader('Loan Eligibility')
            if loan_eligibility == 'Eligible':
                st.markdown(f'<p style="color:green; font-weight:bold;">Based on your bank statement, you are {loan_eligibility} for a loan.</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:red; font-weight:bold;">Based on your bank statement, you are {loan_eligibility} for a loan.</p>', unsafe_allow_html=True)
            
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
            
            if abs_category_dict:
                plot_df = pd.DataFrame(list(abs_category_dict.items()), columns=['Category', 'Amount'])
                
                fig_pie_filtered_category = px.pie(plot_df, values='Amount', names='Category', 
                                                   title='Expense Distribution by Categories',
                                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie_filtered_category)
            else:
                st.write("No data available to plot.")
            
            fig_pie_description = px.pie(df, values='Amount', names='Description', title='Expense Distribution by Description')
            st.plotly_chart(fig_pie_description)

            fig_line = px.line(df, x='Date', y='Amount', title='Daily Expense Trend')
            st.plotly_chart(fig_line)

if __name__ == '__main__':
    main()
