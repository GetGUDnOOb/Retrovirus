import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Define the function for the viral dynamics model
def viral_dynamics_model(lambda_T, d_T, beta, d_I, p, d_V, T0, I0, V0):
    # Time span for the simulation
    t_span = (0, 50)  # Time from 0 to 50 days
    t_eval = np.linspace(t_span[0], t_span[1], 100)  # Points to evaluate

    # System of ODEs
    def viral_dynamics(t, y):
        T, I, V = y
        dT_dt = lambda_T - d_T * T - beta * T * V
        dI_dt = beta * T * V - d_I * I
        dV_dt = p * I - d_V * V
        return [dT_dt, dI_dt, dV_dt]

    # Solve the ODEs
    initial_conditions = [T0, I0, V0]  # Initial populations of T, I, and V
    solution = solve_ivp(viral_dynamics, t_span, initial_conditions, t_eval=t_eval)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(solution.t, solution.y[0], label='Target Cells (T)', color='blue')
    plt.plot(solution.t, solution.y[1], label='Infected Cells (I)', color='red')
    plt.plot(solution.t, solution.y[2], label='Virus Particles (V)', color='green')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('Dynamics of Retroviral Replication in an Infected Host')
    plt.legend()
    plt.grid()

    # Save the plot as an image
    plot_path = os.path.join('static', 'viral_dynamics.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve parameters from the form
        lambda_T = float(request.form['lambda_T'])
        d_T = float(request.form['d_T'])
        beta = float(request.form['beta'])
        d_I = float(request.form['d_I'])
        p = float(request.form['p'])
        d_V = float(request.form['d_V'])
        T0 = float(request.form['T0'])
        I0 = float(request.form['I0'])
        V0 = float(request.form['V0'])

        # Run the simulation
        viral_dynamics_model(lambda_T, d_T, beta, d_I, p, d_V, T0, I0, V0)

        # Redirect to results page
        return redirect(url_for('results'))

    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)

