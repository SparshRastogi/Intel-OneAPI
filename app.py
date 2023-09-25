# Import necessary libraries
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from urllib.request import urlopen
import json
import plotly.graph_objects as go

# Sample data for demonstration (replace with your own data)
data = pd.read_csv(
    'https://raw.githubusercontent.com/SparshRastogi/Covid-19-Risk-Calculator/main/Cases%202021%20Month9.csv')  # Load your data
original = pd.read_csv('https://raw.githubusercontent.com/SparshRastogi/Intel-OneAPI/main/Original.csv')
predicted = pd.read_csv('https://raw.githubusercontent.com/SparshRastogi/Intel-OneAPI/main/Predictions.csv')
corr = pd.read_csv('https://raw.githubusercontent.com/SparshRastogi/Intel-OneAPI/main/Prevaccination%20Correlation.csv')
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df = data
df.rename(columns={'fips': 'FIPS'}, inplace=True)
df['FIPS'] = df['FIPS'].astype('float')
df['FIPS'] = df['FIPS'].astype('int')

# Create a Dash web application
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/slate/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-pzjw8f+ua7Kw1TIq0v8FqFjcJ6pajs/rfdfs3SO+kHO5W5X3Df5E5/5u7f5UJFJ+6',
        'crossorigin': 'anonymous'
    }
])

df['FIPS'] = df['FIPS'].apply(lambda x: '{0:0>5}'.format(x))

choropleth_fig = px.choropleth(df, geojson=counties, locations='FIPS', color='cases',
                               color_continuous_scale=px.colors.sequential.Plasma,
                               range_color=(df['cases'].min(), df['cases'].max()),
                               scope="usa")

choropleth_fig.update_layout(margin=dict(l=20, r=0, b=0, t=70, pad=0), paper_bgcolor="black", plot_bgcolor="black", height=700,
                             title_text='Supervision of daily Covid-19 cases constituency wise', font=dict(color='white'))

# Define the layout of the dashboard
app.layout = html.Div([
    html.Div([
        html.Label("Select State", style={'color': 'white', 'font-family': 'Arial, sans-serif'}),
        dcc.Dropdown(
            id='state-dropdown',
            options=[
                {'label': state, 'value': state} for state in data['state'].unique()
            ],
            value=data['state'].iloc[0],  # Set the initial value to the first state in the dataset
            style={'color': 'black', 'font-family': 'Arial, sans-serif'}
        ),
        html.Label("Select County", style={'color': 'white', 'font-family': 'Arial, sans-serif'}),
        dcc.Dropdown(id='county-dropdown', style={'color': 'black', 'font-family': 'Arial, sans-serif'})
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'background-color': 'black'}),

    html.Div([
        # Static choropleth map for the entire country
        dcc.Graph(
            id='choropleth-map',
            figure=choropleth_fig  # Use the pre-defined choropleth figure
        )
    ], style={'width': '50%', 'display': 'inline-block', 'float': 'right', 'background-color': 'black'}),

    html.Div([
        # Line graph and bar chart together in a new div
        html.Div([
            dcc.Graph(id='line-graph'),
            dcc.Graph(id='bar-chart')
        ], style={'width': '100%', 'background-color': 'black', 'display': 'flex', 'flex-direction': 'column'})
    ], style={'width': '50%', 'display': 'inline-block', 'background-color': 'black'})
])

# Define callback functions to update the county dropdown options
@app.callback(
    Output('county-dropdown', 'options'),
    Input('state-dropdown', 'value')
)
def update_county_dropdown(selected_state):
    # Filter the data for the selected state
    state_data = data[data['state'] == selected_state]

    # Create dropdown options for counties in the selected state
    county_options = [{'label': county, 'value': county} for county in state_data['county'].unique()]

    return county_options


# Define a callback function to set the default value of the county dropdown
@app.callback(
    Output('county-dropdown', 'value'),
    Input('county-dropdown', 'options')
)
def set_default_county_value(county_options):
    # Set the initial value of the county dropdown to the first option
    return county_options[0]['value']


# Define callback functions to update the line graph and bar chart
@app.callback(
    [Output('line-graph', 'figure'),
     Output('bar-chart', 'figure')],
    [Input('state-dropdown', 'value'),
     Input('county-dropdown', 'value')]
)
def update_graphs(selected_state, selected_county):
    # Filter the data for the selected state and county
    predictions = predicted[(predicted['state'] == selected_state) & (predicted['county'] == selected_county)]
    actual = original[(original['state'] == selected_state) & (original['county'] == selected_county)]
    county_data = actual.merge(predictions, on='date')

    # Create the line graph
    line_fig = px.line(county_data, x='date', y=['Actual', 'Predictions'], title='Line Graph')
    rec = corr[(corr['state'] == selected_state) & (corr['county'] == selected_county)]
    # record = rec
    rec.drop(['Unnamed: 0','FIPS', 'state', 'county'], inplace=True, axis=1)
    rec.reset_index(inplace=True, drop=True)
    # Create the bar chart
    bar_fig = go.Figure(data=[go.Bar(y=list(rec.iloc[0].values), x=['retail_and_recreation_percent_change_from_baseline',
                                                                'grocery_and_pharmacy_percent_change_from_baseline',
                                                                'parks_percent_change_from_baseline',
                                                                'transit_stations_percent_change_from_baseline',
                                                                'workplaces_percent_change_from_baseline',
                                                                'residential_percent_change_from_baseline'])])
    return line_fig, bar_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
