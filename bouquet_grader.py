import pandas as pd
from PIL import Image, ImageOps
from dash import Dash, html, dcc, Input, Output

from ex1.learn_flowers_uni_class import learn_flowers_uni_class

app = Dash(__name__)

model, accuracy = learn_flowers_uni_class()

# Create the layout with an upload area and a label
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Table(id='output-data-upload')
], style={'width': '100%', 'height': '100%'})


# Callback to display the uploaded file names
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'filename'),
              prevent_initial_call=True)
def display_uploaded_file_names(list_of_names):
    image_features = pd.read_csv('labels.csv', dtype={'Image Path': 'string', 'Pink': 'float', 'White': 'float', 'Red': 'float',
                                                'Orange': 'float', 'Yellow': 'float', 'Total Flowers': 'float'}, index_col=0)
    image_paths = [f"bouquets/all_images/{name}" for name in list_of_names if f"bouquets/all_images/{name}" in image_features.index]
    image_features = image_features.loc[image_paths, :]
    graded_images = {image_path: model.predict_single(image_features.loc[image_path] / image_features.loc[image_path].sum()) for image_path in image_paths}
    return [
        # Create a centered table row with the image and the grade
        html.Tr([
            html.Td(html.Img(src=ImageOps.contain(Image.open(image_path), (500, 500))), style={'width': '50%'}),
            html.Td(f"Grade: {grade*100:.2f}", style={'text-align': 'center', 'vertical-align': 'middle', 'width': '50%'})
        ], style={'text-align': 'center', 'vertical-align': 'middle', 'padding': '10px', 'width': '100%'})
        for image_path, grade in graded_images.items()
    ]


if __name__ == '__main__':
    app.run_server(debug=True)
