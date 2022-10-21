import plotly.graph_objects as go
import numpy as np
import scipy.io as sio
import torch
import dash
import dash_core_components as dcc
import dash_html_components as html

import argparse

def main(args):
    verts = np.load(args.vert_path)
    faces = np.load(args.faces_path)

    x, y, z = verts.T   
    i, j, k = faces.T

    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='blue', opacity=0.50)])

    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters can be set by -c config.py or by positional params.')

    parser.add_argument('--vert_path', '-v', type=str)
    parser.add_argument('--faces_path', '-f', type=str)
    args = parser.parse_args()

    main(args)