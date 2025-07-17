# Environmental Performance Index (EPI) Analysis

This project provides an interactive analysis of the Environmental Performance Index (EPI) using decision tree modeling. It helps explore how different environmental indicators relate to a country's overall EPI score and classifies countries into performance categories (Low/Medium/High).

## Features

- **Interactive Decision Tree**: Visual representation of how environmental indicators predict EPI classes
- **Searchable Country Data**: Complete dataset with filtering and sorting capabilities
- **Performance Metrics**: Model accuracy and prediction results
- **Responsive Design**: Works well on different screen sizes

## Requirements

- R (>= 4.0.0)
- R packages: tidyverse, rpart, rpart.plot, DT
- R Markdown (for report generation)

## Installation

1. Clone this repository
2. Install required R packages:
   ```R
   install.packages(c("tidyverse", "rpart", "rpart.plot", "DT", "rmarkdown"))
   ```

## Usage

1. Run the analysis:
   ```bash
   Rscript -e "rmarkdown::render('analysis_report.Rmd')"
   ```
2. Open `analysis_report.html` in your web browser to view the interactive report

## Project Structure

- `analysis_report.Rmd`: Main analysis file with decision tree model and visualizations
- `Data/epi2024_data.csv`: Dataset containing country-level environmental indicators and EPI scores
- `biodiversity_analysis_clean.R`: Supporting R script with data processing functions
- `LICENSE`: Apache 2.0 license file

## How to Use the Interactive Report

1. **Explore the Decision Tree**: Understand how environmental indicators predict EPI classes
2. **Search and Filter**: Use the search box and column filters to find specific countries
3. **Sort Data**: Click on column headers to sort the country data
4. **Compare Predictions**: Examine actual vs. predicted EPI classes

## Data Sources

- EPI 2024 Dataset
- Environmental indicators include:
  - BDH: Biodiversity & Habitat
  - ECS: Ecosystem Services
  - FSH: Fish Stocks
  - APO: Air Pollution
  - AGR: Agriculture

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any suggestions or bug reports.
