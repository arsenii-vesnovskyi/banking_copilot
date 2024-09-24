import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objs as go
import graphviz
import streamlit as st
import cohere
from fpdf import FPDF
from datetime import datetime, timedelta
import calendar

### Tabs:
# 1. Log-In
# 2. Dashboard 
#   - Account balance (green if +, red if -)
#   - Last 10 transactions
#   - Upcoming bills for the next 10 days (predicted recurring bills)
#   - Link to the chat tab to get the answers to financial questions
# 3. Transactions with all of the raw data 
#   - transactional data 
#   - filtering: by date, by amount, by source
#   - export possibility: CSV and PDF
# 4. Financial Analytics
# 5. Chat

#############################################
# APP STRUCTURE
#############################################
def main():
    # If the user is not logged in, display the login tab
    if 'tab' not in st.session_state:
        st.session_state['tab'] = 'Log-In'

    # Display the app navigation sidebar
    st.sidebar.title("Navigation")

    # Set the default tab to the last selected tab
    tab = st.sidebar.radio("Go to", ("Log-In", "Dashboard", "Transactions", "Financial Analytics", "Chat", "Process"), 
                           index=("Log-In", "Dashboard", "Transactions", "Financial Analytics", "Chat", "Process").index(st.session_state['tab']))

    # Display the selected tab
    st.session_state['tab'] = tab

    # Display the content of the selected tab based on the user's login status
    if tab == "Log-In":
        tab_login()
    elif tab == "Process":
        tab_process()
    else:
        # Check if the user is logged in, if not, display a message to log in first
        if 'df' not in st.session_state:
            st.markdown(
                """
                <style>
                .centered {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 80vh;
                }
                .message-container {
                    background-color: #FFDDC1;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }
                .message-container h4 {
                    color: #8B0000;
                    margin-bottom: 20px;
                }
                .button-container {
                    display: flex;
                    justify-content: center;
                }
                </style>
                <div class="centered">
                    <div class="message-container">
                        <h4>Please log in (i.e., select a user profile) first to use the Banking Copilot.</h4>
                        <div class="button-container">
                """, 
                unsafe_allow_html=True
            )
            # Display the log-in button
            if st.button('Log-In (click twice)'):
                st.session_state['tab'] = 'Log-In'
            st.markdown(
                """
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # If the user is logged in, display the selected tab
        elif tab == "Dashboard":
            tab_dashboard(st.session_state['df'])
        elif tab == "Transactions":
            tab_transactions(st.session_state['df'])
        elif tab == "Financial Analytics":
            tab_financial_analytics(st.session_state['df'])
        elif tab == "Chat":
            tab_chat(st.session_state['df'])

#############################################
# LOG-IN TAB
#############################################
def tab_login():
    st.image("./assets/login_banner.jpg", use_column_width=True)
    st.title("Welcome to Your Personal Banking Copilot")
    st.write("To access your personalized banking dashboard, please select your user profile from the dropdown menu and click 'Load Data'. This will load your transaction data and allow you to use all the features of the banking copilot.")

    st.markdown("### Select User")  # Use markdown for the label inside the box

    # Display the user selection dropdown
    user = st.selectbox("", ["Select User", "Student", "Worker", "Retiree"], index=0)

    st.markdown('</div>', unsafe_allow_html=True)

    # Load the data for the selected user
    if st.button("Load Data"):
        if user != "Select User":
            df = load_data(user)
            if df is not None:
                st.session_state['df'] = df
                st.session_state['user'] = user
                st.success(f"Logged into {user} account successfully!")
            else:
                st.error("Failed to load data. Please try again.")
        else:
            st.error("Please select a user profile.")

#############################################
# DATA LOADING
#############################################

# Load the data based on the selected user profile
@st.cache_data
def load_data(user):
    """Load the data based on the selected user profile.

    Args:
        user (str): The selected user profile.

    Returns:
        pd.DataFrame: The loaded data.
    """

    if user == "Student":
        df = pd.read_csv("../01-Data/synthetic_data_student_v5.csv")
    elif user == "Worker":
        df = pd.read_csv("../01-Data/synthetic_data_worker_v5.csv")
    elif user == "Retiree":
        df = pd.read_csv("../01-Data/synthetic_data_retiree_v5.csv")
    else:
        df = None

    # Preprocess the data
    if df is not None:
        # Standardize column names
        if 'Amount' in df.columns:
            df.rename(columns={"Amount": "Amount_EUR"}, inplace=True)
        if 'Amount.EUR.' in df.columns:
            df.rename(columns={"Amount.EUR.": "Amount_EUR"}, inplace=True)

        # Sort df by date from the most recent to the oldest
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date', ascending=False)

        # Reindex the dataframe
        df.reset_index(drop=True, inplace=True)
    
    return df

#############################################
# DASHBOARD TAB
#############################################

# Function to calculate financial metrics
def calculate_metrics(df):
    """Calculate financial metrics based on the transaction data.
    
    Args:
        df (pd.DataFrame): The transaction data.
        
    Returns:
        dict: A dictionary containing the calculated financial metrics.
    """
    # Calculate the current date and the dates for the last 3, 6, and 12 months
    today = datetime.today()
    three_months_ago = today - timedelta(days=90)
    six_months_ago = today - timedelta(days=180)
    twelve_months_ago = today - timedelta(days=365)

    # Initialize the metrics dictionary
    metrics = {}
    
    # Income calculations
    metrics['income_last_3_months'] = df[(df['Date'] >= three_months_ago) & (df['Date'] < today) & (df['Amount_EUR'] > 0)]['Amount_EUR'].sum()
    metrics['income_last_6_months'] = df[(df['Date'] >= six_months_ago) & (df['Date'] < today) & (df['Amount_EUR'] > 0)]['Amount_EUR'].sum()
    metrics['income_last_12_months'] = df[(df['Date'] >= twelve_months_ago) & (df['Date'] < today) & (df['Amount_EUR'] > 0)]['Amount_EUR'].sum()
    
    # Expense calculations
    metrics['expenses_last_3_months'] = df[(df['Date'] >= three_months_ago) & (df['Date'] < today) & (df['Amount_EUR'] < 0)]['Amount_EUR'].sum()
    metrics['expenses_last_6_months'] = df[(df['Date'] >= six_months_ago) & (df['Date'] < today) & (df['Amount_EUR'] < 0)]['Amount_EUR'].sum()
    metrics['expenses_last_12_months'] = df[(df['Date'] >= twelve_months_ago) & (df['Date'] < today) & (df['Amount_EUR'] < 0)]['Amount_EUR'].sum()
    
    
    # Current account balance
    metrics['current_account_balance'] = df['Amount_EUR'].sum()
    print(metrics)
    return metrics

# Function to display the home tab
def tab_dashboard(df):
    """Display the home tab."""
    st.image("./assets/AI-Banking.jpg", use_column_width=True)
    st.subheader("Your Personal Banking Copilot")
    st.write(
        "This Banking Copilot helps you extract valuable insights from your past transactions and additionally offers visualizations of your financials. "
        "Exemplary questions may be answered: "
    )
    st.write("- How much money did I spend on groceries last month?")
    st.write("- How much money did I spend on rent last year?")
    st.write("- How much money did I spend on clothes last quarter?")
    st.write("- Can I afford a new Porsche?")
    st.write(
        "The app sources data from your bank account and uses state-of-the-art machine learning and LLM models to serve you to the best way possible."
    )
    
    # Calculate financial metrics
    metrics = calculate_metrics(df)
    
    # Display account balance
    st.subheader("Account Balance")
    account_balance = metrics['current_account_balance']

    balance_color = "#004d00" if account_balance >= 0 else "#8B0000"  # Dark green with less intensity
    st.markdown(
        f"""
        <div style="background-color: {balance_color}; padding: 10px; border-radius: 5px; text-align: center; width: 100%; max-width: 800px; margin: 0 auto;">
            <h4 style="color: white; font-size: 18px;">Your current account balance is: €{account_balance:,.2f}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display recent transactions
    st.subheader("Recent Transactions")
    st.write("Here you can find a preview of your recent transactions.")
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("See More"):
        st.session_state['tab'] = 'Transactions'
        st.experimental_rerun()

    # Display predicted recurring bills for the next 10 days
    st.subheader("Predicted Recurring Bills for the Next 10 Days")
    today = datetime.today()
    recurring_bills = predict_recurring_bills(df, today, 10)

    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    if recurring_bills.empty:
        st.write("No predicted recurring bills in the next 10 days.")
    else:
        st.dataframe(recurring_bills)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Go to Chat for Financial Questions"):
        st.session_state['tab'] = 'Chat'
        st.experimental_rerun()

# Function to predict recurring bills
def predict_recurring_bills(df, today, days):
    """
    The bill prediction identifies recurring transactions by payee and purpose that occur 
    on multiple days of the month and estimates their next occurrence within the next 
    10 days from the current date, ensuring no repetition of payees.

    Args:
    df (pd.DataFrame): The transaction data.
    today (datetime): The current date.
    days (int): The number of days to predict recurring bills.

    Returns:
    pd.DataFrame: A DataFrame containing the predicted recurring bills.
    """
    # Extract day of month for each transaction
    df['Day'] = df['Date'].dt.day
    
    # Group by Payee and Purpose to find recurring transactions
    recurring_bills = []
    grouped = df.groupby(['Payee', 'Purpose'])
    for (payee, purpose), group in grouped:
        # Find the days of the month on which transactions occur
        days_of_month = group['Day'].value_counts().index.tolist()
        
        # If transactions occur on more than one day of the month, consider it as recurring
        if len(days_of_month) > 1:
            for day in days_of_month:
                try:
                    predicted_date = today.replace(day=day)
                except ValueError:
                    # Skip invalid dates
                    continue
                
                if today < predicted_date <= today + timedelta(days=days):
                    recurring_bills.append({
                        'Payee': payee,
                        'Purpose': purpose,
                        'Predicted Date': predicted_date,
                        'Amount_EUR': group['Amount_EUR'].mean()  # Assuming average amount for prediction
                    })
    
    if recurring_bills:
        # Convert to DataFrame and drop duplicates to unify recurring bills from the same payee and purpose
        recurring_bills_df = pd.DataFrame(recurring_bills)
        recurring_bills_df = recurring_bills_df.sort_values(by='Predicted Date').drop_duplicates(subset=['Payee', 'Purpose'])
        return recurring_bills_df
    else:
        return pd.DataFrame(columns=['Payee', 'Purpose', 'Predicted Date', 'Amount_EUR'])

#############################################
# TRANSACTIONS TAB
#############################################

# Function to generate a PDF from the filtered transactions
def generate_pdf(dataframe):
    pdf = FPDF(orientation='L', unit='mm', format='A4')  # Ensure orientation is set to landscape
    pdf.add_page()
    
    # Add a title
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Filtered Transactions", ln=True, align='C')
    
    # Add a table header
    pdf.set_font("Arial", size=10)
    col_width = (pdf.w - 20) / len(dataframe.columns)  # Adjust column width to fit all columns in the page
    row_height = pdf.font_size * 1.5
    for column in dataframe.columns:
        pdf.cell(col_width, row_height, txt=column, border=1)
    pdf.ln(row_height)
    
    # Add data to the table
    for row in dataframe.itertuples():
        for cell in row[1:]:
            pdf.cell(col_width, row_height, txt=str(cell), border=1)
        pdf.ln(row_height)
    
    return pdf.output(dest='S').encode('latin1')

# Function to display the transactions tab
def tab_transactions(df):
    """Display all transactions with filtering options."""
    st.subheader("Transactions")
    st.write("Here you can view all of your transactions.")

    # Filtering by time range
    st.markdown("### Filter by Date Range")
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    start_date, end_date = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filtering by transaction amount
    st.markdown("### Filter by Transaction Amount")
    min_amount = float(df['Amount_EUR'].min())
    max_amount = float(df['Amount_EUR'].max())
    amount_range = st.slider("Select amount range", min_value=min_amount, max_value=max_amount, value=(min_amount, max_amount))

    # Filtering by transaction source
    st.markdown("### Filter by Transaction Source")
    sources = df['Purpose'].unique().tolist()
    selected_sources = st.multiselect("Select transaction sources", sources, default=sources)

    # Apply filters
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    filtered_df = filtered_df[(filtered_df['Amount_EUR'] >= amount_range[0]) & (filtered_df['Amount_EUR'] <= amount_range[1])]
    if selected_sources:
        filtered_df = filtered_df[filtered_df['Purpose'].isin(selected_sources)]

    st.dataframe(filtered_df, height=600, width=1000)
    
    # Option to export filtered data as a CSV file
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download transactions as CSV",
        data=csv,
        file_name='filtered_transactions.csv',
        mime='text/csv',
    )

    # Option to export filtered data as a PDF file
    pdf = generate_pdf(filtered_df)
    st.download_button(
        label="Download transactions as PDF",
        data=pdf,
        file_name='filtered_transactions.pdf',
        mime='application/pdf',
    )

#############################################
# FINANCIAL ANALYTICS TAB
#############################################

# Function to display financial analytics
def tab_financial_analytics(df):
    """Display financial analytics."""
    st.subheader("Financial Analytics")
    st.write("Here you can find insightful visualizations of your financial patterns.")
    
    # Plotly boxplot
    st.subheader("Spending by Category")
    fig = px.box(df, x='Purpose', y='Amount_EUR', title="Distribution of Amounts by Purpose")
    st.plotly_chart(fig)
    
    # Graphviz plot
    st.subheader("Transaction Flow")

    df_limited = df.head(15)
    df_summarized = df_limited.groupby(['Payer', 'Payee'])['Amount_EUR'].sum().reset_index()
    
    dot = graphviz.Digraph()
    # Set graph attributes
    dot.attr(rankdir='LR', size='10,10', ratio='fill', bgcolor='#f9f9f9')

    # Set default node attributes
    dot.attr('node', shape='box', style='filled', fillcolor='#1f77b4', fontcolor='white', fontsize='10', fontname='Arial')

    # Set default edge attributes
    dot.attr('edge', color='#1f77b4', style='solid', fontsize='10', fontname='Arial')
    
    dot.attr(size='8,10', ratio='fill', splines='true', overlap='false', rankdir='LR', ranksep='1.5') 
    for _, row in df_summarized.dropna(subset=['Payer', 'Payee', 'Amount_EUR']).iterrows():
        dot.node(row['Payer'], row['Payer'], fillcolor="darkgreen")
        dot.node(row['Payee'], row['Payee'], fillcolor="lightcoral")
        dot.edge(row['Payer'], row['Payee'], label=f"{row['Amount_EUR']} EUR")
    st.graphviz_chart(dot)

    # Pydeck plot
    st.subheader("Geographical Distribution of Transactions")
    df_map = df.copy()
    df_map['latitude'] = np.random.uniform(low=48.0, high=50.0, size=(df_map.shape[0],))
    df_map['longitude'] = np.random.uniform(low=10.0, high=12.0, size=(df_map.shape[0],))
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[longitude, latitude]',
        get_color='[200, 30, 0, 160]',
        get_radius=200,
    )
    view_state = pdk.ViewState(
        latitude=df_map['latitude'].mean(),
        longitude=df_map['longitude'].mean(),
        zoom=6,
        pitch=50,
    )
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.pydeck_chart(r)

#############################################
# CHAT TAB
#############################################

# co = cohere.Client('KqGGX6p9yHgo46Hy70fHJRmfYDZbQ56gVGysOVc6')  # Simon's Free Trial API key
co = cohere.Client("O65u993SeF15EVD4o240Vt6BCufNKnyw0AEDksQB") # Arsenii's Free Trial API key

#############################################
# Function for the tool selection
#############################################

# Function to select the appropriate tool based on the user's question
def select_tool(question):
    """Select the appropriate tool based on the user's question.

    Args:
        question (str): The user's question.
    
    Returns:
        str: The response from the LLM with or without indication which tool to use.
    """
    # Prepare a prompt that guides the LLM to format responses as required
    prompt = """You are a banking copilot tool. You are designed to help customers with their financial questions.\n""" +\
             """If the question is not related to finance, explain that you can only assist with financial queries.\n""" +\
             """For financial questions, provide the best answer you can.\n""" +\
             """In four specific cases, structure the response starting with the tool name in capital letters.\n\n""" +\
             """Here are the specific cases where you should return a response consisting only of the tool name and the provided user question:\n\n""" +\
             """1. If the question is related to the customer's transaction history  and past transactions, start your response with the word TRANSACTION and then provide user request exactly as you received it.\n""" +\
             """2. If the request is about making a transfer of money, start your response with the word TRANSFER and then provide user request exactly as you received it.\n""" +\
             """3. If the request is about affording something specific, start your response with the word AFFORD and then provide user request exactly as you received it.\n""" +\
             """4. If the request is explicitly asking to visualize financial data, start your response with the word VISUALIZATION and then provide user request exactly as you received it.\n\n""" +\
             """If the question does not fall into one of these four cases, provide a concise and polite answer that does NOT mention how you function internally.\n\n""" +\
             f"""Customer's question: {question}"""

    # Call LLM or any other model with the provided prompt
    response = co.chat(message=prompt)

    # Log response for debugging
    print(f"Response from select_tool: {response.text}")

    return response.text if response.text else None

#############################################
# Functions for the first tool: TRANSACTION
#############################################

# Function to retrieve Python statement from Coherence AI
def get_python_statement(question, data):
    """Get the Python statement to extract information from the transaction data based on the user's question.

    Args:
        question (str): The user's question.
        data (pd.DataFrame): The transaction data.

    Returns:
        str: The Python statement to extract information from the data.
    """
    # Extract column names and datatypes from the dataframe
    columns = data.columns
    dtypes = data.dtypes
    # Create a list of column names and their types
    columns_with_types = [f"{col} (dtype: {dtype})" for col, dtype in zip(columns, dtypes)]

    # Identify the owner of the bank account
    owner = data[data['Sales_Type'] == 'Outgoing']['Payer'].values[0]
    owner_IBAN = data[data['Payer'] == owner]['Payer_IBAN'].values[0]

    # Identify categorical columns and extract unique values
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    unique_values = {col: data[col].unique().tolist() for col in categorical_columns}

    # Prepare message for Cohere AI, including column names and user input
    prompt = f"""You are a banking copilot tool. You have access to a pandas dataframe called "data" with the following columns: {', '.join(columns_with_types)}.\n"""
    # Add unique values from categorical columns to the prompt
    for col, values in unique_values.items():
        prompt += f"The column '{col}' has the following unique values: {', '.join(map(str, values))}.\n"

    prompt += f"""Provide the python statement that will get the information from that dataframe that is needed to answer the user's question.\n""" + \
        f"""Remember is that in outgoing payments, the Payer is the owner of the bank account and the ingoing payemnts (getting money) the owner is the Payee.\n""" + \
        """Also remember that for any spending the Sales_Type is 'Outgoing'.\n""" + \
        f"""User is the owner of the bank account with the name '{owner}'.\n""" + \
        f"""That means that for any questions about spending the owner is the Payer, and for any questions about ingoing payments (e.g., sources of money), the owner is the Payee. The indices of the dataframe are also just numbers.\n""" + \
        f"""Only return the one-liner python code or an empty string in case the user's question is unrelated to the financial topics or cannot be answered with the data available.\n""" + \
        f"""Your code will be saved as variable python_statement and integrated in the application using following line of code: result = eval(python_statement). So please make sure your response is only consist of a single python statement which is formatted properly to be interpreted with that function.\n""" + \
        f"""Do NOT assign the python statement to any variables in the prompt.\n""" + \
        f"""User's question: {question}"""
    
    # Call Cohere API to get Python statement
    response = co.chat(message = prompt + question)
    
    # Extract Python statement from the response
    extracted_code = response.text.split("python")[-1].replace('\n', ' ').strip().rstrip('`') if response.text else None
    print(f"Extracted Python statement: {extracted_code}")
    return extracted_code if extracted_code else None

# Function to generate response using Python statement and provided data
def generate_response(python_statement, data, question, metrics):
    """Generate a response using the Python statement and provided data based on the user's question.

    Args:
        python_statement (str): The Python statement to extract information from the data.
        data (pd.DataFrame): The transaction data.
        question (str): The user's question.
        metrics (dict): The financial metrics that can be used to answer the question.

    Returns:
        str: The response to the user's question based on the extracted information.
    """
    # Extract column names from the dataframe
    columns = data.columns.tolist()

    # Prepare a prompt with the extracted information and the user's question
    # If the Python statement is empty, return a message indicating that the question cannot be answered
    if not python_statement:
        return "Oops, it seems that the question cannot be answered with the data available."
    try:
        # Apply Python statement to the dataframe
        result = eval(python_statement)
        print(f"Result of python code evaluation: {result}")  # Log result for debugging
        # Provide the information to the LLM and get the response
        prompt_with_data = f"""You are a useful banking copilot tool. You have access to a pandas dataframe with the following columns: {', '.join([f'"{column_name}"' for column_name in columns])}.\n""" + \
            f"You have obtained the following question from the user: {question}\n" + \
            f"From the dataframe you were able to extract the following information using python statement: {result}.\n" + \
            f"Please provide the answer to the user's question based on the extracted information, if you find that the information is useful for answering the question." + \
            f"Here are some financial metrics you might find useful: {metrics}.\n" + \
            "However, do not mention the python statement itslef. Be confident in your response." + \
            f"Otherwise, if you cannot answer question with the retrieved data, or if the question is unrelated to financial topics, provide a polite response stating so." 
        response = co.chat(message = prompt_with_data)
        # print(f"Prompt with data: {prompt_with_data}")  # Log prompt with data for debugging
        # print(f"Final response: {response.text}")  # Log final response for debugging
        return response.text
    except Exception as e:
        # Print full traceback and error message if Python statement execution fails
        print(e)
        return f"Oops, there was an error executing Python statement: {str(e)}"
    
#############################################
# Functions for the second tool: TRANSFER
#############################################

# Function to get the unique payers, payees, and their IBANs from the transaction dataframe columns 'Payer', 'Payee', and 'IBAN'
# and return them as a dataframe with columns 'Name' and 'IBAN'
def get_unique_contacts(df):
    """Get unique contacts from the transaction data.
    
    Args:
        df (pd.DataFrame): The transaction data.
        
    Returns:
        pd.DataFrame: A DataFrame containing unique contacts and their IBANs."""
    
    # Get unique payers, payees, and their IBANs
    unique_payers = df['Payer'].unique()
    unique_payees = df['Payee'].unique()
    unique_contacts = np.unique(np.concatenate((unique_payers, unique_payees), axis=0))
    unique_contacts_df = pd.DataFrame(unique_contacts, columns=['Name'])
    unique_contacts_df['IBAN'] = unique_contacts_df['Name'].apply(
        lambda x: df[df['Payer'] == x]['Payer_IBAN'].values[0] if x in unique_payers else (
            df[df['Payee'] == x]['Payee_IBAN'].values[0] if x in unique_payees else None))
    return unique_contacts_df

# Function to get transaction details from the user request
def get_the_transaction_details(user_request):
    """Get the transaction details from the user's request.
    
    Args:
        user_request (str): The user's request.
        
    Returns:
        dict: A dictionary containing the extracted transaction details."""

    prompt = """You are a banking copilot tool. Your only task is to extract two things from the user's request: 
    the name of the contact the user wants to transfer money to and the amount they want to transfer.\n
    You should only return the name and amount in form of a Python dictionary and nothing else. For example, if the user's request is 
    'I want to transfer money to Arsenii Vesnovskyi', you should return: {"name": "Arsenii Vesnovskyi", "amount": None}. 
    If the user's request is 'I want to send 10 euros to Marta Peregrin', you should return: {"name": "Marta Peregrin", "amount": 10}. 
    If you think there is no name in the request that is provided to you, store the value None under the key "name".
    If the currency is not EUR, you should convert it to EUR. If the amount is not provided, you should store None under the key "amount" in the dictionary.\n""" +\
    f"""User's request: {user_request}"""
    
    response = co.chat(message=prompt)
    
    return eval(response.text) if response.text else None

# Function to search for the contact in the unique_contacts_df. Capitalize the first letter of the name and surname.
def search_contact(contact_name, unique_contacts_df):
    """Search for the contact in the unique contacts DataFrame.

    Args:
        contact_name (str): The name of the contact.
        unique_contacts_df (pd.DataFrame): The DataFrame containing unique contacts and their IBANs.

    Returns:
        pd.DataFrame: A DataFrame containing the contact details if found, otherwise None.
    """
    # If contact name is not provided, return None
    if contact_name is None:
        return None
    
    # Capitalize the first letter of the name and surname
    contact_name = contact_name.title()

    # Search for the contact in the unique contacts DataFrame
    # If the contact is found, return the contact details
    if contact_name in unique_contacts_df['Name'].values:
        return unique_contacts_df[unique_contacts_df['Name'] == contact_name]
    else:
        return None
    
# Function to get the IBAN of the contact
def get_IBAN(contact_name, unique_contacts_df):
    """Get the IBAN of the contact.

    Args:
        contact_name (str): The name of the contact.
        unique_contacts_df (pd.DataFrame): The DataFrame containing unique contacts and their IBANs.

    Returns:
        str: The IBAN of the contact if found, otherwise None.
    """
    # If contact name is not provided, return None
    if contact_name is None:
        return None
    
    # Search for the contact in the unique contacts DataFrame
    contact = search_contact(contact_name, unique_contacts_df)
    if contact is not None:
        return contact['IBAN'].values[0]
    else:
        return None
    
# Function to check and confirm the transaction details
# Define the button to check and confirm the transaction and update the transaction dataframe and the underlying csv file.
# The check includes that all fields are filled in,
# the amount is not negative, and the name is not the owner of the bank account.
def check_and_confirm_transaction(contact_name, amount, IBAN, owner):
    """Check and confirm the transaction details. 
    Save the transaction to the transaction dataframe and the underlying csv file.

    Args:
        contact_name (str): The name of the contact.
        amount (float): The amount to transfer.
        IBAN (str): The IBAN of the contact.
        owner (str): The owner of the bank account.
    
    Returns:
        None
    """
    #print("Entering check_and_confirm_transaction...")  # Debug statement
    if contact_name and amount is not None and IBAN:
        if amount > 0:
            if contact_name != owner: 
                df = st.session_state['df']
                new_transaction = {
                    'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Payer': owner,
                    'Payer_IBAN': df[df['Payer'] == owner]['IBAN'].values[0],
                    'Payee_IBAN': IBAN,
                    'Payee': contact_name,
                    'Purpose': 'Personal',
                    'Salestype': 'Outgoing',
                    'Amount_EUR': -amount
                }
                # Check the session state variable for the selected user for login
                user = st.session_state['user']
                # Use the original df here instead of new_df
                df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)
                df.to_csv('../01-Data/synthetic_data_'+str(user).lower() + '_.csv', index=False)
                st.session_state['df'] = df
                #print(new_transaction)
                st.write('Transaction confirmed!')
                reset_transaction_state()
            else:
                st.write('You cannot transfer money to yourself!')
        else:
            st.write('Amount must be positive!')
    else:
        st.write('All fields must be filled in!')

# Function to reset session state variables related to transaction
def reset_transaction_state():
    st.session_state['transaction_details'] = None
    st.session_state['form_display'] = False
    
# Define editable pop-up Streamlit form to display the transaction details
def transaction_form(contact_name, amount, IBAN, owner):
    """Display the transaction form to confirm the transaction details.
    Trigger the check_and_confirm_transaction function when the form is submitted.

    Args:
        contact_name (str): The name of the contact.
        amount (float): The amount to transfer.
        IBAN (str): The IBAN of the contact.
        owner (str): The owner of the bank account.
    
    Returns:
        None
    """
    with st.form(key='transaction_form'):
        contact_name_input = st.text_input('Contact Name', value=contact_name, key='contact_name_input')
        amount_input = st.number_input('Amount (EUR)', value=amount if amount else 0, key='amount_input')
        IBAN_input = st.text_input('IBAN', value=IBAN if IBAN else "", key='iban_input')

        submit_button = st.form_submit_button('Confirm Transaction')
        if submit_button:
            st.write("Form submission detected!")
            check_and_confirm_transaction(contact_name_input, amount_input, IBAN_input, owner=owner)

# Define the function that puts all the pieces together: takes user's request as an input,
# extracts the transaction details, searches for the contact in the unique_contacts_df,
# displays the transaction form.
def make_transaction(user_request, unique_contacts_df):
    """Make a transaction based on the user's request.

    Args:
        user_request (str): The user's request.
        unique_contacts_df (pd.DataFrame): The DataFrame containing unique contacts and their IBANs.
    
    Returns:
        None
    """
    transaction_details = get_the_transaction_details(user_request)
    if transaction_details:
        # Handle None value for transaction_details['name']
        contact_name = transaction_details.get('name')
        if contact_name:
            IBAN = get_IBAN(contact_name = contact_name, unique_contacts_df=unique_contacts_df)
            # print(f"IBAN: {IBAN}, Contact Name: {contact_name}")
            transaction_details['iban'] = IBAN
            st.session_state['transaction_details'] = transaction_details  # Store transaction details in session state
            st.session_state['form_display'] = True
        
        else:
            st.write('Contact name could not be extracted.')
    else:
        st.write('Could not extract the transaction details!')


#############################################
# Functions for the third tool: AFFORD
#############################################

def get_affordability_details(user_request):
    """Get the item and amount from the user's request to check affordability.
    
    Args:
        user_request (str): The user's request.
        
    Returns:
        dict: A dictionary containing the extracted item and amount.
    """
    prompt = """You are a banking copilot tool. Your only task is to extract the item and the amount the user wants to know if they can afford. 
    You should only return the item and amount in the form of a Python dictionary and nothing else. For example, if the user's request is 
    'Can I afford to buy a new laptop for 1000 euros?', you should return: {"item": "laptop", "amount": 1000}. 
    If the user's request is 'Can I afford to rent an apartment that costs 800 euros per month?', you should return: {"item": "apartment", "amount": 800}. 
    If you think there is no item or amount in the request, store the value None under the key "item" or "amount". 
    If the currency is not EUR, you should convert it to EUR.\n""" +\
    f"""User's request: {user_request}"""
    
    response = co.chat(message=prompt)
    
    return eval(response.text) if response.text else None

# Function to perform a web search for the price of an item
def get_item_price_from_web(item):
    """Perform a web search to get the price of an item.

    Args:
        item (str): The item for which the price is to be searched.

    Returns:
        str: The result of the websearch by LLM.
    """
    message = f"What is the average price of a {item} in euros?"
    response = co.chat(
        message=message,
        connectors=[{"id": "web-search"}]
    )
    
    # Extract price information from the response
    price_text = response.text
    
    return price_text

# Function to check if the user can afford the item
def can_afford(item, amount, metrics):
    """Check if the user can afford the item based on the amount and financial metrics.

    Args:
        item (str): The item the user wants to buy.
        amount (str or float): The amount of the item.
        metrics (dict): The financial metrics of the user.
    
    Returns:
        str: The response to the user's question.
    """

    if isinstance(amount, str):
        prompt = f"""You are a useful banking copilot tool. Your main ability is answering the affordability questions.""" +\
        f"""You have access to the following financial metrics of the user: {metrics}.\n""" +\
        f"""The user asked if they can afford a {item}.""" +\
        f"""Based on the web search, we received the following information: {amount}.""" +\
        f"""Please provide the answer to the user's question based on the extracted information.""" +\
        f"""If you cannot answer the question with the retrieved data, or if the question is unrelated to financial topics, provide a polite response stating so."""
    elif isinstance(amount, (int, float)):
        prompt = f"""You are a useful banking copilot tool. Your main ability is answering the affordability questions.""" +\
        f"""You have access to the following financial metrics of the user: {metrics}.\n""" +\
        f"""The user asked if they can afford a {item}.""" +\
        f"""Based on the web search, we received the following information: {amount}.""" +\
        f"""Please provide the answer to the user's question based on the extracted information.""" +\
        f"""If you cannot answer the question with the retrieved data, or if the question is unrelated to financial topics, provide a polite response stating so."""
    else:
        return "I'm sorry, I couldn't understand your request. Please try asking in a different way."
    
    response = co.chat(message=prompt)
    return response.text if response.text else None

# Main function to handle affordability questions
def handle_affordability_question(user_request, df):
    """Handle the user's affordability question by uusing the functions defined above.

    Args:
        user_request (str): The user's request.
        df (pd.DataFrame): The transaction data.
    
    Returns:
        str: The response to the user's question.
    """

    # Extract the item and amount from the user's request
    details = get_affordability_details(user_request)
    
    if not details:
        return "I'm sorry, I couldn't understand your request. Please try asking in a different way."
    
    item = details.get('item')
    amount = details.get('amount')
    
    if amount is None:
        # Perform a web search to get the price of the item
        amount = get_item_price_from_web(item)
        if amount is None:
            return "I'm sorry, I couldn't find the price of the item. Please try asking in a different way."
    
    # Calculate financial metrics
    metrics = calculate_metrics(df)
    
    # Determine if the user can afford the item
    return can_afford(item, amount, metrics)




#############################################
# Functions for the fourth tool: VISUALIZE
#############################################

def process_input(input_text, df):
    """Process the user's input to extract the relevant information for visualization and create the visualization.

    Args:
        input_text (str): The user's input text.
        df (pd.DataFrame): The transaction data.
    
    Returns:
        fig: plotly figure (or the response to the user's question if not related to visualization, or an error message if an error occurs)
    """
    # Filter all categories with unique() from df["Purpose"]
    df_categories = df["Purpose"].unique()
    df_years = df['Date'].dt.year.unique()
    
    spending_words = ['spendings', 'expenses', 'expenditure', 'spending', 'expense', 'expenditures', "how", "change"]
    lower_input_text = input_text.lower()
    
    if any(word in lower_input_text for word in spending_words):
        if "invest" in lower_input_text:
            return "Unfortunately, I cannot visualize the investments for you yet. Coming soon."
        else:
            try:
                # Use Cohere API to extract category and year from input text
                prompt = f"""The available categories are: {', '.join(df_categories)}. The available years are: {df_years}""" + \
                        f"Additionally, there are two sales types: 'Ingoing', 'Outgoing'. If the query involves spendings or expenses, consider the 'Outgoing' sales type. " + \
                        f"For questions about savings, all of the above-mentioned categories should be taken into account. Moreover, both 'Ingoing' and 'Outgoing' sales types are relevant in such case." + \
                        f"When the query is about income, apply the 'Ingoing' sales type exclusively. Now, based on the following text, extract the relevant categories, years, and sales types: '{input_text}' " + \
                        f"Please provide the extracted information as a python dictionary (as a string) with the keys 'Categories', 'Years', 'Sales Types'. The values of the dictionary should always be python lists, even if they only contain one value." + \
                        f"If multiple categories or sales types are found, the lists will contain multiple string values. The type of data in the list for years should be numeric." + \
                        "Example (Multiple categories or sales types): {'Categories': ['Travel', 'Food'], 'Years': [2022, 2023], 'Sales Types' : ['Ingoing', 'Outgoing']}. """+ \
                        "Second Example (Single categories or sales types): {'Categories': ['Travel'], 'Years': [2022], Sales Types: ['Ingoing']}." + \
                        "Please only return the python dictionary as a response. Do not assign it to any variables in your response."
                
                response = co.chat(message=prompt)
                # Ensure the response contains text and the expected text field
                if not hasattr(response, 'text'):
                    return "Received an empty or invalid response from the API."
                
                extracted_code = response.text.split("python")[-1].replace('\n', ' ').strip().rstrip('`') if response.text else None
                # st.write("Cohere response:", extracted_code)  # Debugging                
                
                output = extracted_code.strip()
                # st.write("API output:", output)  # Debugging

                fig_dict = eval(output)
                
                # Initialize variables
                category = fig_dict.get('Categories', None)
                year = fig_dict.get('Years', None)
                sales_type = fig_dict.get('Sales Types', None)

                if category is None or year is None or sales_type is None:
                    st.write("Could not extract category, year, or sales type from your question.")
                    return

                # st.write(f"Extracted Category: {category}, Year: {year}, Sales Type: {sales_type}")  # Debugging
                
                # Validate extracted values
                if not all(c in df['Purpose'].unique() for c in category):
                    return f"One or more categories '{category}' not found in data."
                if not all(y in df['Date'].dt.year.unique() for y in year):
                    return f"One or more years '{year}' not found in data."
                if not all(s in df['Sales_Type'].unique() for s in sales_type):
                    return f"One or more sales types '{sales_type}' not found in data."

                # Determine the type of the question
                if 'Outgoing' in sales_type and 'Ingoing' not in sales_type:
                    q_type = "Spendings"
                elif 'Ingoing' in sales_type and 'Outgoing' not in sales_type:
                    q_type = "Income"
                    return "Sorry, I am not able to visualize your income yet. Coming soon."
                else:
                    q_type = "Savings"
                    return "Sorry, I am not able to visualize your savings yet. Coming soon."
                    
                # Filter data for the specified category and year
                category_data = df[(df['Purpose'].isin(category)) & (df['Date'].dt.year.isin(year)) & (df['Sales_Type'].isin(sales_type))]
                # st.write(category_data)  # Debugging
                
                if category_data.empty:
                    return f"No data available for category '{category}' in year '{year}'."

                # Group data by month and calculate total spendings
                category_data['Month'] = category_data['Date'].dt.month.astype(int)
                category_data['Year'] = category_data['Date'].dt.year.astype(int)
                
                month_mapping = {i: calendar.month_abbr[i] for i in range(1, 13)}
                
                category_data['Month'] = category_data['Month'].map(month_mapping).astype(str)
                
                # st.write(category_data) # Debugging
                
                # Group data by month and calculate total spendings
                category_data = category_data.groupby(['Year', 'Month'])['Amount_EUR'].sum().reset_index()
                
                # Create a DataFrame with all month-year combinations
                years = category_data['Year'].unique()
                all_months = pd.DataFrame([(year, month) for year in years for month in calendar.month_abbr[1:13]],
                                        columns=['Year', 'Month'])

                # Merge with the original data to fill in missing months
                category_data = pd.merge(all_months, category_data, on=['Year', 'Month'], how='left').fillna(0)

                # Convert Month column to a categorical type with the correct order
                category_data['Month'] = pd.Categorical(category_data['Month'], categories=calendar.month_abbr[1:13], ordered=True)

                # Sort the DataFrame by Year and Month
                category_data = category_data.sort_values(by=['Year', 'Month'])

                # st.write(category_data.columns) # Debugging

                # Revert the negative values to positive in "Amount_EUR" column if it is "Spendings"
                if q_type == "Spendings":
                    category_data['Amount_EUR'] = category_data['Amount_EUR'].abs()
                
                # st.write("Monthly data:", category_data) # Debugging

                try:
                    fig = px.line(category_data, x='Month', y='Amount_EUR', color='Year',
                                  title=f'{", ".join(category)} {q_type} in {", ".join(map(str, year))}',
                                  labels={'Amount_EUR': f'Total {q_type}', 'Month': 'Month'})
                    fig.update_layout(xaxis_type='category')
                    # fig.update_xaxes(title_text='Month')
                    # fig.update_yaxes(title_text=f'Total {q_type}')
                    return fig
                except Exception as e:
                    return f"Failed to create a valid Plotly figure: {e}"
            
            except Exception as e:
                return f"An error occurred: {e}"
            
    else:
        return "I am sorry, I am currently only able to visualize your spendings. Please ask about your expenditures on the catrgories of interest in a specific time-frame."

#############################################
# Function to display the chat tab interface
#############################################

def tab_chat(df):
    """Display chat interface."""

    # Identify the owner of the bank account (which would be the value of the Payer column for transactions with negative amounts)
    owner = df[df['Sales_Type'] == 'Outgoing']['Payer'].values[0]
    # print(f"Owner of the bank account: {owner}")
    owner_IBAN = df[df['Payer'] == owner]['Payer_IBAN'].values[0]
    # print(f"Owner's IBAN: {owner_IBAN}")

    # Calculate financial metrics as they are used in the prompts
    metrics = calculate_metrics(df)

    # Get unique contacts from the transaction dataframe
    unique_contacts_df = get_unique_contacts(df)

    st.subheader("Chat with Banking Copilot")
    user_input = st.text_input("Ask a question:")
    
    ask_button = st.button("Ask Copilot")

    if ask_button and user_input:
        with st.spinner('Analyzing data and loading...'):
            # Call select_tool to get the response
            tool_response = select_tool(user_input)

            # Check if tool_response indicates a specific tool being used
            if tool_response:
                tool_names = ["TRANSACTION", "TRANSFER", "AFFORD", "VISUALIZATION"]
                tool_used = next((name for name in tool_names if name.upper() in tool_response), None)

                if tool_used == "TRANSACTION":
                    st.write(f"The tool {tool_used} is being used.")
                    # Call the transaction tool functions
                    python_statement = get_python_statement(question=user_input, data=df)
                    response = generate_response(python_statement=python_statement, 
                                                 data=df, 
                                                 question=user_input, 
                                                 metrics=metrics)

                    # Display the response from the transaction tool
                    st.write(response)
                elif tool_used == "TRANSFER":
                    # Initialize session state variables if not present
                    if 'form_display' not in st.session_state:
                        st.session_state.form_display = False
                    if 'transaction_details' not in st.session_state:
                        st.session_state.transaction_details = None
                    
                    st.write(f"The tool {tool_used} is being used.")
                    
                    reset_transaction_state()  # Reset state before making a new transaction
                    make_transaction(user_request=user_input, unique_contacts_df=unique_contacts_df)

                    # Display the form if the form_display flag is True
                    if st.session_state.form_display:
                        transaction_form(
                            st.session_state.transaction_details['name'],
                            st.session_state.transaction_details['amount'],
                            st.session_state.transaction_details['iban'],
                            owner=owner
                            )
                elif tool_used == "AFFORD":
                    # Call the affordability tool functions
                    st.write(f"The tool {tool_used} is being used.")
                    response = handle_affordability_question(user_input, df)
                    st.write(response)

                elif tool_used == "VISUALIZATION":
                    # Call the visualization tool functions
                    st.write(f"The tool {tool_used} is being used.")
                    response = process_input(user_input, df)
                    try:
                        if isinstance(response, go.Figure):
                            st.plotly_chart(response, selection_mode = "points")
                        elif isinstance(response, str):
                            st.write(response)
                        else:
                            st.write("Oops! Something went wrong. Please try again.")
                    except Exception as e:
                        st.write(f"An error occurred while plotting the chart: {e}. Please let my creators know and try again.")
                elif tool_used:
                    # Display message indicating the tool being used
                    st.write(f"The tool {tool_used} is currently in development. We will notify you when you can use it.")
                else:
                    # Display the general tool response
                    st.write(tool_response)
            else:
                st.write("Could not process the request at the moment. Please try again.")

#############################################
# PROCESS TAB
#############################################
# Function to display the process tab where we describe our journey to the prototype
def tab_process():
    st.title("Our Journey to the Prototype")

    # Timeline container 1
    st.markdown("---")
    st.markdown("## Question \U00002753")
    st.markdown("### Key question: Can we build a natural language banking copilot using Streamlit?")

    st.image("./assets/question_1.jpg", use_column_width=True)
    
    st.markdown("---")
    
    st.markdown("""
    ## Goals \U0001F3AF
    ### Goal 1: Reliability & Focus
    - High reliability in returning desired responses
    - Few focused use-cases
    
    ### Goal 2: Flexibility to User Needs
    - High flexibility in terms of identifying user requests
    - Using LLM for automatic allocation of the use case
    
    ### Goal 3: Intuitive User Interface
    - Quick understanding of prototype functionality
    - Proper guidance without excessive text
    """)

    # Timeline container 2
    st.markdown("---")
    st.markdown("## Approach \U0001F4A1")

    st.markdown("""
    ### Iterative
    - Very basic prototype as a starting point
    - Gradual addition of features
    
    ### Flexible
    - Quick (in)validation of ideas
    - Continuous testing and improvement
    
    ### Focused
    - Brainstorming of multiple potential ideas
    - Narrowed focus to the most promising use-cases
    """)

    st.image("./assets/agile.gif", use_column_width=True)

    # Timeline container 3
    st.markdown("---")
    st.markdown("## Iterations \U0001F4BB")

    st.markdown("""
    ### Structure Decision
    - Separation of the tool into pre-defined content and dynamic chat interactions
    
    ### First Iteration - Single Call
    - Single call approach using Cohere on 150 data rows
    - Inconsistent and unreliable responses
    """)

    st.image("./assets/first_iteration.jpg", use_column_width=True)

    st.markdown("""
    ### Second Iteration - Python Statement Integration
    - Generate Python statements through LLM, as numerical data not suitable for embeddings
    - More reliable calculations, occasional malfunctions
    - Substantial improvements through more info about data structure in the prompt
    - No need to send all the user's data to LLM
    """)

    st.image("./assets/second_iteration.jpg", use_column_width=True)

    st.markdown("""
    ### Third Iteration - Tool Integration with Extra Knowledge Base
    - Define tools with an additional knowledge base (e.g., precalculated metrics, web-search)
    - The tools are: TRANSACTION, TRANSFER, AFFORD, VISUALIZATION
    - Implement an overarching LLM to select the needed tool
    - Overarching LLM implemented by hand, as more traceable for debugging
    """)

    st.image("./assets/third_iteration.jpg", use_column_width=True)

    st.markdown("---")
    st.markdown("## Learnings \U0001F4DA")

    st.markdown("""
    ### Streamlit Reactivity and Session Management
    - Consistent reloading implies the need for session variables.
    - Complexity increases with the number of consecutive functions and user interactions.
    
    ### Prompt Management
    - Effective prompt management is extremely powerful.
    - Handling all possible user inputs is challenging.
    
    ### Interaction Simplification
    - Multiple LLMs calls complicates the creation of chat history
    - Interaction limited to one question - one answer allows for testing multiple use-cases
    - Full-scale chat could be a future development
    
    ### Multiple LLM Calls
    - Using multiple LLM calls in each session increases the complexity and run-time
    - Usage of paid API calls is inevitable for the full-scale application
    """)
    st.image("./assets/monkey.gif", use_column_width=True)


################################################################
# ENSURE THAT APP RUNS ONLY WHEN THE SCRIPT IS EXECUTED DIRECTLY
################################################################
if __name__ == "__main__":
    main()