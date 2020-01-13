import plotly.graph_objs as go
import pandas as pd

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands': 'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Palau': 'PW',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}


def main():
    df = pd.read_csv('joined_results_1.csv')
    df = df.append(pd.read_csv('joined_results_2.csv'))
    df = df.append(pd.read_csv('joined_results_rec_vt.csv'))
    df = df.append(pd.read_csv('joined_results_auth_hi.csv'))

    df['recursive_state'] = df['recursive_state'].map(us_state_abbrev)
    df['authoritative_state'] = df['authoritative_state'].map(us_state_abbrev)

    recursive_locations = df.drop_duplicates(subset=['recursive_latitude', 'recursive_longitude'])
    authoritative_locations = df.drop_duplicates(subset=['authoritative_latitude', 'authoritative_longitude'])

    recursive_agg = df.groupby(['recursive_state'], as_index=False).median()
    # auth_agg = df.groupby(['authoritative_state'], as_index=False).median()

    fig = go.Figure(data=[go.Choropleth(
        locations=recursive_agg['recursive_state'],  # Spatial coordinates
        z=recursive_agg['rtt'].astype(float),  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Reds',
        colorbar_title="Median RTT",
    ),
        go.Scattergeo(lon=recursive_locations['recursive_longitude'],
                      lat=recursive_locations['recursive_latitude'],
                      mode='markers',
                      marker_color='blue',
                      marker_size=7,
                      hoverinfo="none",
                      locationmode='USA-states',
                      showlegend=False),
        go.Scattergeo(lon=authoritative_locations['authoritative_longitude'],
                      lat=authoritative_locations['authoritative_latitude'],
                      mode='markers',
                      marker_color='yellow',
                      hoverinfo="none",
                      locationmode='USA-states',
                      showlegend=False)
    ])

    fig.update_layout(
        title_text='Testing!',
        geo_scope='usa',  # limit map scope to USA
    )

    fig.write_image("test.svg", scale=10)


if __name__ == "__main__":
    main()
