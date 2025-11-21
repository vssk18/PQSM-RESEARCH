.PHONY: all clean data plots report

all: data plots report
	@echo "Complete pipeline finished"

data:
	python src/generate_experimental_data.py

plots: data
	python generate_all_plots.py
	python src/analysis/advanced_statistics.py

report: plots
	python make_report_html.py

clean:
	rm -f analysis/analysis/*.csv
	rm -f analysis/analysis/plots/*.png
	rm -f analysis/report.html

help:
	@echo "Available targets:"
	@echo "  make all    - Run complete pipeline"
	@echo "  make data   - Generate experimental data"
	@echo "  make plots  - Create all visualizations"
	@echo "  make report - Build HTML report"
	@echo "  make clean  - Remove generated files"
