# Environmental Performance Index (EPI) Prediction Model

This project uses machine learning to predict a country's Environmental Performance Index (EPI) class (Low/Medium/High) based on key environmental indicators. The model helps identify which factors most significantly impact a country's overall environmental performance.

## Features

- **Decision Tree Model**: Predicts EPI class using environmental indicators
- **Interactive Analysis**: Includes code for training, evaluating, and visualizing the model
- **Future Scenario Predictions**: Simulates how different environmental conditions affect EPI scores
- **HTML Report**: Generates a detailed, self-contained report with visualizations

## Requirements

- R (>= 4.0.0)
- R packages: tidyverse, caret, rpart, rpart.plot
- R Markdown (for report generation)

## Usage

1. Clone this repository
2. Install required R packages: `install.packages(c("tidyverse", "caret", "rpart", "rpart.plot", "rmarkdown"))`
3. Run the analysis: `Rscript -e "rmarkdown::render('analysis_report.Rmd')"`
4. Open `analysis_report.html` to view the results

## Project Structure

- `analysis_report.Rmd`: Main R Markdown file containing the analysis
- `Data/epi2024_data.csv`: Input data file with environmental indicators
- `analysis_output/`: Directory containing model outputs and logs
- `biodiversity_analysis_clean.R`: Supporting R script with data processing functions

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
