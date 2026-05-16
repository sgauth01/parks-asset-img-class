[![Apache 2.0 License](https://img.shields.io/github/license/bcgov/nr-epd-aq-statements.svg)](/LICENSE) [![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](<Redirect-URL>)

# parks-asset-img-class

## Project overview

This repository contains early work for a 2026 UBC MDS capstone project on using image analysis to classify BC Parks infrastructure assets and predict attributes such as asset type, material, railing presence, size ranges, structure position, and number of steps.

At the moment, the main runnable artifact is a Quarto report:

- `reports/Image_analysis_of_park_infrastructure_report.qmd`

Rendering this file creates both:

- `reports/Image_analysis_of_park_infrastructure_report.html`
- `reports/Image_analysis_of_park_infrastructure_report.pdf`

## Repository structure

```text
.
├── environment.yml
├── notebooks/
│   └── data_exploration.ipynb
└── reports/
    ├── Image_analysis_of_park_infrastructure_report.qmd
    └── references.bib
```

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate bcparks_capstone
```

The report is built with [Quarto](https://quarto.org/). Install Quarto if it is not already available:

```bash
quarto --version
```

PDF rendering also requires a LaTeX installation. If PDF rendering fails because LaTeX is missing, install TinyTeX with:

```bash
quarto install tinytex
```

## Render the report

From the repository root, run:

```bash
quarto render reports/Image_analysis_of_park_infrastructure_report.qmd
```

This command renders all formats listed in the report YAML, currently HTML and PDF.

To render only one format:

```bash
quarto render reports/Image_analysis_of_park_infrastructure_report.qmd --to html
quarto render reports/Image_analysis_of_park_infrastructure_report.qmd --to pdf
```

## Experiment tracking with MLflow

All model runs are tracked with [MLflow](https://mlflow.org/). The
default tracking store is a local file directory at `./mlruns` (gitignored),
so no server is required and nothing leaves the machine.

End-to-end smoke test (synthetic data only, no SharePoint download needed):

```bash
python scripts/mlflow_smoke_test.py
```

This fits a scikit-learn dummy classifier and regressor on synthetic data,
and writes both runs to `./mlruns` under the experiment
**`parks-asset-img-class`**.

View the runs in your browser:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Run the real baseline models with cross-validation on the processed training
data:

```bash
python scripts/run_baseline.py
```

This writes `results/baseline_classification_results.csv` and
`results/baseline_classification_cv_folds.csv`, and logs one MLflow run per
attribute when `mlflow` is installed. Categorical attributes and binned
continuous attributes such as `length_bin`, `width_bin`, `fall_height_bin`, and
`steps_bin` use a majority-class predictor with cross-validated `accuracy`,
`macro_f1`, and `weighted_f1`.

Run the unit tests:

```bash
pytest -q tests/test_baseline.py tests/test_mlflow_utils.py
```

## Current status

This project is in an early exploratory stage. The report currently describes the project motivation, research question, dataset assumptions, data challenges, and a proposed modelling approach. The notebook directory is available for exploration work as the project develops.

## Getting Help or Reporting an Issue

To report bugs/issues/feature requests, please file an [issue](https://github.com/bcgov/parks-asset-img-class/issues/new).

## How to Contribute

If you would like to contribute, please see our [CONTRIBUTING](CONTRIBUTING.md) guidelines.

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## License

    Copyright 2026 Province of British Columbia

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and limitations under the License.


------------------------------------------------------------------------

*This project was created using the [bcgovr](https://github.com/bcgov/bcgovr) package.*
