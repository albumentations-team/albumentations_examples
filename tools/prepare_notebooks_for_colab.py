import argparse
import os

import nbformat as nbf


parser = argparse.ArgumentParser()
parser.add_argument("--input-directory")
parser.add_argument("--output-directory")

markdown = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


colab_cells = [
    markdown("## Prepare the Google Colab environment"),
    markdown("#### Download images"),
    markdown(
        "Download images that are used in the notebook and save to the `images` folder in the Colab environment."
    ),
    code(
        '''!wget -q https://github.com/albumentations-team/albumentations_examples/archive/master.zip -O /tmp/albumentations_examples.zip
!unzip -o -qq /tmp/albumentations_examples.zip -d /tmp/albumentations_examples
!cp -r /tmp/albumentations_examples/albumentations_examples-master/notebooks/images .
!echo "Images are successfully downloaded"'''
    ),
    markdown("#### Install the latest version of Albumentations"),
    markdown(
        "Google Colab has an outdated version of Albumentations so we will install the latest stable version from PyPi."
    ),
    code(
        '''!pip install -q -U albumentations
!echo "$(pip freeze | grep albumentations) is successfully installed"'''
    ),
    markdown("## Run the example"),
]


def is_markdown_cell(cell):
    return cell["cell_type"] == "markdown"


def is_header_cell(cell):
    return cell["source"].startswith("#")


def get_first_non_introduction_cell_index(cells):
    if not (is_markdown_cell(cells[0]) and is_header_cell(cells[0])):
        return 0

    if is_markdown_cell(cells[1]) and is_header_cell(cells[1]):
        return 1

    return 2


if __name__ == "__main__":
    args = parser.parse_args()
    notebook_files = [
        f for f in os.listdir(args.input_directory) if f.endswith(".ipynb")
    ]
    for notebook_file in notebook_files:
        filepath = os.path.join(args.input_directory, notebook_file)
        notebook = nbf.read(filepath, as_version=4)
        index = get_first_non_introduction_cell_index((notebook["cells"]))
        notebook["cells"] = (
            notebook["cells"][:index] + colab_cells + notebook["cells"][index:]
        )
        nbf.write(notebook, os.path.join(args.output_directory, notebook_file))
