import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
from scipy.stats import norm

# ==============================================================
# Plotting Helpers
# ==============================================================


def save_histogram_with_fit(df, column, filename):
    """
    Saves a histogram with an overlaid fitted normal distribution.
    """

    values = df[column]
    mu, sigma = norm.fit(values)

    plt.figure(figsize=(6, 3.2))

    # Histogram
    count, bins, _ = plt.hist(values, bins=25, density=True, alpha=0.6)

    # Normal fit curve
    x = np.linspace(bins[0], bins[-1], 300)
    pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, linewidth=2)

    plt.title(f"Histogram + Normal Fit: {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_spc_chart(df, column, filename):
    """
    Create an X̄ control chart (individual measurements).
    """

    values = df[column]
    mean = values.mean()
    sigma = values.std()

    ucl = mean + 3 * sigma
    lcl = mean - 3 * sigma

    plt.figure(figsize=(6, 3.2))
    plt.plot(values.index, values, marker="o", linestyle="-")
    plt.axhline(mean, color="black", linestyle="--", label="Mean")
    plt.axhline(ucl, color="red", linestyle="--")
    plt.axhline(lcl, color="red", linestyle="--")

    plt.title(f"SPC Chart (X̄): {column}")
    plt.xlabel("Measurement Index")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ==============================================================
# Main Program
# ==============================================================


def main():

    # ----------------------------------------------------------
    # 1. Create dataset
    # ----------------------------------------------------------
    np.random.seed(42)
    n = 200

    data = pd.DataFrame(
        {
            "Diameter_mm": np.random.normal(50.0, 0.02, n),
            "Length_mm": np.random.normal(100.0, 0.05, n),
            "Thickness_mm": np.random.normal(5.0, 0.01, n),
        }
    )

    csv_path = "part_repeatability_dataset.csv"
    data.to_csv(csv_path, index=False)

    # ----------------------------------------------------------
    # 2. Prepare PDF
    # ----------------------------------------------------------
    pdf_path = "part_variability_report.pdf"
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter, title="Part Dimension Repeatability Report"
    )

    story = []

    # Title page
    story.append(
        Paragraph("<b>Part Dimension Repeatability Report</b>", styles["Title"])
    )
    story.append(Spacer(1, 0.4 * inch))

    intro = """
    This report presents statistical analyses for three dimensional features.
    For each feature, the page includes:

    • Summary Statistics  
    • Histogram with Normal Distribution Fit  
    • SPC Chart (X̄)  

    All graphs for a given feature appear on the same page for clarity.
    """
    story.append(Paragraph(intro, styles["BodyText"]))
    story.append(PageBreak())

    # ----------------------------------------------------------
    # 3. One page per feature, with all plots on the page
    # ----------------------------------------------------------
    for column in data.columns:

        # Summary statistics
        summary = data[column].describe()

        # Paths for saved figures
        hist_path = f"{column}_hist_fit.png"
        spc_path = f"{column}_spc.png"

        # Create plots
        save_histogram_with_fit(data, column, hist_path)
        save_spc_chart(data, column, spc_path)

        # Header
        story.append(Paragraph(f"<b>{column}</b>", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))

        # Summary section
        stats_text = f"""
        <b>Summary Statistics</b><br/>
        Mean: {summary['mean']:.6f}<br/>
        Standard Deviation: {summary['std']:.6f}<br/>
        Min: {summary['min']:.6f}<br/>
        Max: {summary['max']:.6f}<br/>
        25th Percentile: {summary['25%']:.6f}<br/>
        Median: {summary['50%']:.6f}<br/>
        75th Percentile: {summary['75%']:.6f}<br/><br/>
        """
        story.append(Paragraph(stats_text, styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))

        # Histogram with normal fit
        story.append(Paragraph("<b>Histogram with Normal Fit</b>", styles["Heading2"]))
        story.append(Image(hist_path, width=6.2 * inch, height=3.5 * inch))
        story.append(Spacer(1, 0.3 * inch))

        # SPC Chart
        story.append(Paragraph("<b>SPC Chart (X̄)</b>", styles["Heading2"]))
        story.append(Image(spc_path, width=6.2 * inch, height=3.5 * inch))
        story.append(Spacer(1, 0.3 * inch))

        # New page for next feature
        story.append(PageBreak())

    # Build PDF
    doc.build(story)

    print("Dataset saved to:", csv_path)
    print("PDF saved to:", pdf_path)


# ==============================================================
# Script Entry Point
# ==============================================================

if __name__ == "__main__":
    main()
