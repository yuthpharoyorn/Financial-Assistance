from flask import Flask, render_template, request
from flask_httpauth import HTTPBasicAuth
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import io
import base64

app = Flask(__name__)
auth = HTTPBasicAuth()

users = {
    "admin": "123"
}

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username
    return None


file_path = 'personal finance data 2.xlsx'  
data = pd.read_excel(file_path)
data['Date / Time'] = pd.to_datetime(data['Date / Time'])
data = data.dropna()


data['Month'] = data['Date / Time'].dt.month
data['Quarter'] = data['Date / Time'].dt.quarter
data['Rolling_Expense_3M'] = data['Debit/Credit'].rolling(window=3).mean()


def plot_to_img(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


 
@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')
from datetime import datetime, timedelta



import plotly.express as px

@app.route('/expenses', methods=['GET', 'POST'])
@auth.login_required
def expenses():

    current_date = datetime.now()

    # Default to last month if no filter is selected
    date_filter = request.args.get('filter', 'last_month')

    # Filter data based on the selected period
    if date_filter == 'last_month':
        start_date = current_date.replace(day=1) - timedelta(days=1)
        start_date = start_date.replace(day=1)
        end_date = start_date.replace(day=28) + timedelta(days=4)
        end_date = end_date.replace(day=1) - timedelta(days=1)
        filtered_data = data[(data['Date / Time'] >= start_date) & (data['Date / Time'] <= end_date)]
    elif date_filter == 'last_3_months':
        start_date = current_date - timedelta(days=90)
        end_date = current_date
        filtered_data = data[(data['Date / Time'] >= start_date) & (data['Date / Time'] <= end_date)]
    else:

        filtered_data = data

    # Summarize expenses by category
    expense_summary = filtered_data[filtered_data['Income/Expense'] == 'Expense'].groupby('Category')['Debit/Credit'].sum().reset_index()

    # Summarize income by category
    income_summary = filtered_data[filtered_data['Income/Expense'] == 'Income'].groupby('Category')['Debit/Credit'].sum().reset_index()

    # Create a Plotly bar chart for expenses
    fig_expenses = px.bar(
        expense_summary,
        x='Category',
        y='Debit/Credit',
        title=f"Expenses by Category ({date_filter.replace('_', ' ').title()})",
        text_auto=True,
        color_discrete_sequence=['#e74c3c']  # Red color for expenses
    )

    # Create a Plotly bar chart for income
    fig_income = px.bar(
        income_summary,
        x='Category',
        y='Debit/Credit',
        title=f'Income by Category ({date_filter.replace("_", " ").title()})',
        text_auto=True,
        color_discrete_sequence=['#27ae60']  # Green color for income
    )

    # Adjust layout for both charts
    for fig in [fig_expenses, fig_income]:
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Total Amount",
            template="plotly_dark",
            dragmode=False,  # Disable drag mode
            autosize=True,  # Auto resize the chart
            margin=dict(l=50, r=50, t=50, b=50),  # Add margin
            height=600  # Increase chart height
        )

    # Convert Plotly figures to HTML
    plot_expenses_html = fig_expenses.to_html(full_html=False)
    plot_income_html = fig_income.to_html(full_html=False)

    return render_template(
        'expenses.html',
        plot_expenses_html=plot_expenses_html,
        plot_income_html=plot_income_html,
        date_filter=date_filter
    )



submitted_budgets = {}
submitted_bills = {}

@app.route('/budget', methods=['GET', 'POST'])
@auth.login_required
def budget():
    global submitted_budgets, submitted_bills

    if request.method == 'POST':
        action = request.form.get('action')

        # Add or update a budget
        if action == 'set_budget':
            category = request.form.get('category')
            if category == 'Other':
                category = request.form.get('custom_category')  # Get custom category name
            amount = request.form.get('amount')
            if category and amount:
                submitted_budgets[category] = float(amount)


        elif action == 'delete_budget':
            category = request.form.get('delete_category')
            if category and category in submitted_budgets:
                del submitted_budgets[category]

        # Add or update a bill
        elif action == 'set_bill':
            bill_name = request.form.get('bill_name')
            bill_amount = request.form.get('bill_amount')
            bill_due_date = request.form.get('bill_due_date')
            if bill_name and bill_amount and bill_due_date:
                submitted_bills[bill_name] = {'amount': float(bill_amount), 'due_date': bill_due_date}

        # Delete a bill
        elif action == 'delete_bill':
            bill_name = request.form.get('delete_bill_name')
            if bill_name and bill_name in submitted_bills:
                del submitted_bills[bill_name]

    return render_template(
        'budget1.html',
        submitted_budgets=submitted_budgets,
        submitted_bills=submitted_bills
    )


@app.route('/insights', methods=['GET', 'POST'])
@auth.login_required
def insights():
    global submitted_budgets  # Access submitted budgets globally for comparison

    # Default to 'all_time' if no filter is selected
    date_filter = request.args.get('filter', 'all_time')

    # Get the current date
    current_date = datetime.now()

    # Filter data by date range if provided
    if date_filter == 'last_month':
        # Calculate the start and end dates for the last month
        start_date = current_date.replace(day=1) - timedelta(days=1)
        start_date = start_date.replace(day=1)
        end_date = start_date.replace(day=28) + timedelta(days=4)  # Go to next month
        end_date = end_date.replace(day=1) - timedelta(days=1)
        filtered_data = data[(data['Date / Time'] >= start_date) & (data['Date / Time'] <= end_date)]
    elif date_filter == 'last_3_months':

        three_months_ago = current_date - pd.DateOffset(months=3)
        filtered_data = data[data['Date / Time'] >= three_months_ago]

    else:
        filtered_data = data

    # Calculate total income, total expenses, and savings
    total_income = filtered_data[filtered_data['Income/Expense'] == 'Income']['Debit/Credit'].sum()
    total_expenses = filtered_data[filtered_data['Income/Expense'] == 'Expense']['Debit/Credit'].sum()
    savings = total_income - total_expenses

    # Expense by category calculation
    expense_by_category = (
        filtered_data[filtered_data['Income/Expense'] == 'Expense']
        .groupby('Category')['Debit/Credit']
        .sum().reset_index()
    )

    # Create a pie chart for income vs expenses
    income_expense_summary = (
        filtered_data.groupby('Income/Expense')['Debit/Credit']
        .sum().reset_index()
    )

    fig = px.pie(income_expense_summary, names='Income/Expense', values='Debit/Credit', title='Income vs Expenses')
    plot_html_income_expense = fig.to_html(full_html=False)

    # Financial Health Calculation based on spending
    if total_income > 0:
        spending_percentage = total_expenses / total_income

        if spending_percentage >= 0.8:
            financial_health = "Bad"
        elif spending_percentage <= 0.3:
            financial_health = "Excellent"
        elif spending_percentage <= 0.5:
            financial_health = "Good"
        elif spending_percentage <= 0.7:
            financial_health = "Fair"

    else:
        financial_health = "No income"

    return render_template(
        'insights.html',
        plot_html_income_expense=plot_html_income_expense,
        total_income=total_income,
        total_expenses=total_expenses,
        savings=savings,
        expense_by_category=expense_by_category.to_html(classes='table table-bordered', index=False),
        financial_health=financial_health,
        date_filter=date_filter
    )





@app.route('/advice', methods=['GET', 'POST'])
@auth.login_required
def advice():
    model_filename = 'rf_model.joblib'

    data_ml = data.copy()
    data_ml['Date'] = data_ml['Date / Time'].dt.to_period('M').astype(str)
    data_ml = pd.get_dummies(data_ml, columns=['Category', 'Sub category', 'Mode', 'Income/Expense'])
    data_ml.drop(columns=['Date / Time', 'Date'], inplace=True)
    X = data_ml.drop(columns=['Debit/Credit'])
    y = data_ml['Debit/Credit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    try:
        model = joblib.load(model_filename)
    except FileNotFoundError:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_filename)
    current_date = pd.to_datetime('today')
    next_month = current_date + pd.DateOffset(months=1)
    predicted_expense = 0


    for i in range(30):
        new_data = {
            'Category': 'Category_Name',
            'Sub category': 'Sub_Category_Name',
            'Mode': 'Mode_Type',
            'Income/Expense': 'Expense',
        }
        new_data_df = pd.DataFrame(new_data, index=[0])
        new_data_df = pd.get_dummies(new_data_df, columns=['Category', 'Sub category', 'Mode', 'Income/Expense'])
        new_data_df = new_data_df.reindex(columns=X.columns, fill_value=0)
        predicted_expense += model.predict(new_data_df)[0]


    predicted_expense = round(predicted_expense, 2)

    # Filter the data for the last 3 months
    current_date = pd.to_datetime('today')
    three_months_ago = current_date - pd.DateOffset(months=3)
    last_three_months_data = data[data['Date / Time'] >= three_months_ago]


    total_income = last_three_months_data[last_three_months_data['Income/Expense'] == 'Income']['Debit/Credit'].sum()
    total_expenses = last_three_months_data[last_three_months_data['Income/Expense'] == 'Expense']['Debit/Credit'].sum()

    if total_income > 0:
        spending_percentage = total_expenses / total_income

        if spending_percentage <= 0.2:
            financial_health_last_three_months = "Excellent"
            advice_message = "Great job! You are managing your expenses well."
        elif spending_percentage <= 0.5:
            financial_health_last_three_months = "Good"
            advice_message = "You're doing well, but there's room to cut back."
        elif spending_percentage <= 0.7:
            financial_health_last_three_months = "Fair"
            advice_message = "Consider reviewing your expenses and cutting back where possible."
        else:
            financial_health_last_three_months = "Poor"
            advice_message = "Your spending is high. Focus on reducing unnecessary expenses."
    else:
        financial_health_last_three_months = "No income"
        advice_message = "You don't have income data for the last three months."


    expense_by_category = (
        last_three_months_data[last_three_months_data['Income/Expense'] == 'Expense']
        .groupby('Category')['Debit/Credit']
        .sum().reset_index()
    )
    most_expensive_category = expense_by_category.sort_values(by='Debit/Credit', ascending=False).iloc[0]
    most_expensive_category_name = most_expensive_category['Category']
    most_expensive_category_amount = most_expensive_category['Debit/Credit']

    return render_template('advice.html',
                           predicted_expense=predicted_expense,
                           financial_health_last_three_months=financial_health_last_three_months,
                           advice_message=advice_message,
                           most_expensive_category_name=most_expensive_category_name,
                           most_expensive_category_amount=most_expensive_category_amount)





if __name__ == '__main__':
    app.run(debug=True, port=5002)