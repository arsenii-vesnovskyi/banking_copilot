# Banking Copilot Project

## Overview
This project is a Banking Copilot application developed using Streamlit. The application provides a personalized banking dashboard for users to manage their financial data effectively. Users can view their account balance, recent transactions, upcoming bills, and perform financial analytics.

## Directory Structure
The project directory is structured as follows:

### 00-Info
- Group project - Instructions.pdf: Contains the instructions for the group project.
- questions_test.txt: Test questions for the project.

### 01-Data
- .DS_Store: System file.
- synthetic_data_retiree_v5.csv: Synthetic data file for retirees.
- synthetic_data_student_v5.csv: Synthetic data file for students.
- synthetic_data_worker_v5.csv: Synthetic data file for workers.

### 02-App
- assets - folder with images for the app
- app-final.py: Main application script.

### 03-Data Generation
- .RData: R data file produced during data generation.
- .Rhistory: R history file produced during data generation..
- Synthetic-Data_Retiree-v5.R: R script for generating synthetic data for retirees.
- Synthetic-Data_Student-v5.R: R script for generating synthetic data for students.
- Synthetic-Data_Worker-v5.R: R script for generating synthetic data for workers.

### requirements.txt
- The file with the lobraries required to run the app

## Usage of the App (just in case)
1. Launch the application by running the Streamlit command (streamlit run app-final.py).
2. Select your user profile (Student, Worker, Retiree) from the dropdown menu on the login page.
3. Click "Load Data" to load your transaction data.
4. Navigate through the tabs (Dashboard, Transactions, Financial Analytics, Chat, Process) to explore various features of the application.

## Features
- User Login: Select user profile (Student, Worker, Retiree) to load personalized data.
- Dashboard: View account balance, recent transactions, and upcoming recurring bills.
- Transactions: Filter and export transaction data based on date, amount, and source.
- Financial Analytics: Explore the predefined visualizations of your financial data with various plots and charts.
- Chat: Interact with the banking copilot for financial questions and insights.
- Process: View the journey and development process of the project.

## Notes
- Ensure the CSV files are correctly placed in the `01-Data` directory for data loading.
- Modify file paths in the code if necessary to match your local directory structure.

## Acknowledgements
We thank all team members and professor Jose Rodriguez for the contribution to this project! Special thanks to Streamlit, Plotly, and Cohere for providing the tools and APIs that made this project possible.
