#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a PDF report for the Crop Yield Analysis in India.
This script creates a detailed PDF report with observations and interpretations
of all the charts, graphs, and analysis results.
"""

import os
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# Paths
plots_dir = "../plots"
output_dir = "../output"
report_path = "crop_yield_analysis_report.pdf"

# Load analysis summary
with open(os.path.join(output_dir, "analysis_summary.json"), "r") as f:
    summary = json.load(f)

# Prepare PDF document
doc = SimpleDocTemplate(report_path, pagesize=letter)
styles = getSampleStyleSheet()

# Create custom styles
title_style = ParagraphStyle(
    'Title',
    parent=styles['Title'],
    fontSize=24,
    spaceAfter=30,
    alignment=TA_CENTER
)

heading1_style = ParagraphStyle(
    'Heading1',
    parent=styles['Heading1'],
    fontSize=18,
    spaceAfter=12,
    spaceBefore=24,
)

heading2_style = ParagraphStyle(
    'Heading2',
    parent=styles['Heading2'],
    fontSize=16,
    spaceAfter=10,
    spaceBefore=20,
)

normal_style = ParagraphStyle(
    'Normal',
    parent=styles['Normal'],
    fontSize=12,
    spaceBefore=6,
    spaceAfter=6,
    alignment=TA_JUSTIFY
)

# Add custom caption style
caption_style = ParagraphStyle(
    'Caption',
    parent=styles['Italic'],
    fontSize=10,
    alignment=TA_CENTER
)

# Start building document content
elements = []

# Title
elements.append(Paragraph("Crop Yield Analysis in India", title_style))
elements.append(Spacer(1, 0.25*inch))

# Executive Summary
elements.append(Paragraph("Executive Summary", heading1_style))
elements.append(Paragraph("""
This report presents a comprehensive analysis of crop yields in India based on a dataset containing 
19,689 records spanning from 1997 to 2020. The analysis explores patterns and relationships to identify 
factors that influence crop yields and provides data-driven recommendations for farmers.
""", normal_style))

elements.append(Paragraph("Key Findings:", heading2_style))
findings = [
    "Production shows the strongest correlation with yield (correlation coefficient: 0.57)",
    "Coconut, Sugarcane, and Banana are the highest-yielding crops in India",
    "Five distinct farm clusters were identified based on area, rainfall, fertilizer, and pesticide usage",
    "Different crops perform optimally under specific conditions of rainfall, season, and agricultural inputs"
]
for finding in findings:
    elements.append(Paragraph(f"• {finding}", normal_style))

elements.append(Spacer(1, 0.25*inch))

# 1. Data Exploration
elements.append(Paragraph("1. Data Exploration", heading1_style))
elements.append(Paragraph(f"""
The analysis was conducted on a dataset containing {summary['data_summary']['total_records']} records of 
crop production in India. The dataset includes {summary['data_summary']['unique_crops']} unique crops 
grown across {summary['data_summary']['unique_states']} states during {summary['data_summary']['unique_seasons']} 
different seasons, spanning from {summary['data_summary']['year_range'][0]} to {summary['data_summary']['year_range'][1]}.
""", normal_style))

# 1.1 Yield Distribution
elements.append(Paragraph("1.1 Yield Distribution", heading2_style))
if os.path.exists(os.path.join(plots_dir, "yield_distribution.png")):
    elements.append(Image(os.path.join(plots_dir, "yield_distribution.png"), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure 1: Distribution of crop yields across all data points", caption_style))
    elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph("""
<b>Observations:</b> The yield distribution shows a significant right skew, indicating that most crops have relatively 
low yields, while a small number of crops have exceptionally high yields. Coconut stands out with yields orders of 
magnitude higher than other crops, which reflects its production measurement in nuts per hectare rather than weight. 
The long tail of the distribution suggests that specific high-yielding crops like Coconut, Sugarcane, and Banana 
significantly outperform other crops in terms of yield.
""", normal_style))

# 1.2 Yield by Crop and Season
elements.append(Paragraph("1.2 Yield by Crop and Season", heading2_style))
if os.path.exists(os.path.join(plots_dir, "yield_by_top_crops.png")):
    elements.append(Image(os.path.join(plots_dir, "yield_by_top_crops.png"), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure 2: Yield distribution across top 10 crops", caption_style))
    elements.append(Spacer(1, 0.2*inch))

if os.path.exists(os.path.join(plots_dir, "yield_by_season.png")):
    elements.append(Image(os.path.join(plots_dir, "yield_by_season.png"), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure 3: Yield distribution across different seasons", caption_style))
    elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph("""
<b>Observations on Crop Yields:</b> The boxplot of top 10 crops reveals extreme variation in yields across different 
crop types. Coconut shows the highest median yield by a significant margin, followed by Sugarcane and Banana. These 
three crops consistently outperform others, making them potentially lucrative choices for farmers. The wide range 
and presence of outliers in some crops like Sugarcane indicate that yield can vary considerably depending on growing 
conditions and farming practices.

<b>Observations on Seasonal Yields:</b> The seasonal analysis shows that crops grown during the "Whole Year" tend to 
have higher median yields compared to crops grown in specific seasons. "Summer" crops generally show higher yields 
than "Winter" or "Kharif" (monsoon) season crops, which could be attributed to better growing conditions or the types 
of crops typically grown in each season. This suggests that farmers might benefit from selecting crops that can be 
cultivated throughout the year or during summer when possible.
""", normal_style))

# 1.3 Top Performing Crops
elements.append(Paragraph("1.3 Top Performing Crops", heading2_style))

data = [["Crop", "Average Yield"]]
for crop, yield_value in list(summary["top_crops"]["by_yield"].items())[:5]:
    data.append([crop, f"{yield_value:.2f}"])

table = Table(data, colWidths=[3*inch, 2*inch])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))

elements.append(table)
elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph("""
<b>Observations on Top Crops:</b> The top five crops by average yield show a stark contrast in productivity. Coconut 
leads with an extraordinary average yield of over 8,600 units per hectare, primarily because it's measured in nuts 
rather than weight. Sugarcane follows with approximately 52 tons per hectare, making it one of the most productive 
crops by weight. Banana, Tapioca, and Potato round out the top five with yields of 27, 17, and 13 tons per hectare 
respectively. These high-yielding crops represent potential opportunities for farmers looking to maximize productivity 
per unit of land.
""", normal_style))

# 2. Correlation Analysis
elements.append(Paragraph("2. Correlation Analysis", heading1_style))
elements.append(Paragraph("""
A correlation analysis was performed to identify relationships between different variables and crop yield.
""", normal_style))

if os.path.exists(os.path.join(plots_dir, "correlation_heatmap.png")):
    elements.append(Image(os.path.join(plots_dir, "correlation_heatmap.png"), width=6*inch, height=5*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure 4: Correlation matrix showing relationships between different variables", caption_style))
    elements.append(Spacer(1, 0.2*inch))

if os.path.exists(os.path.join(plots_dir, "yield_correlations.png")):
    elements.append(Image(os.path.join(plots_dir, "yield_correlations.png"), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure 5: Variables correlated with crop yield", caption_style))
    elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph(f"""
<b>Observations on Correlations:</b> The correlation analysis reveals that Production has the strongest positive 
correlation with Yield ({summary['correlation_analysis']['yield_correlations']['Production']:.2f}), which is expected 
since these variables are directly related. Interestingly, Annual_Rainfall shows only a very slight positive correlation 
with Yield ({summary['correlation_analysis']['yield_correlations']['Annual_Rainfall']:.2f}), suggesting that while 
rainfall is important, other factors may play more significant roles in determining crop yield.

The correlations between Yield and input factors like Fertilizer ({summary['correlation_analysis']['yield_correlations']['Fertilizer']:.4f}) 
and Pesticide ({summary['correlation_analysis']['yield_correlations']['Pesticide']:.4f}) are surprisingly weak, 
indicating that simply increasing these inputs may not necessarily improve yields. This could suggest that optimal 
application rather than quantity is more important, or that other unmeasured factors (like soil quality or farming 
techniques) have greater influence.

The correlation matrix also shows strong positive correlations between Area and Fertilizer (0.88) and between Fertilizer 
and Pesticide (0.92), indicating that larger farms tend to use more fertilizer, and farms that use more fertilizer also 
tend to use more pesticide.
""", normal_style))

# 3. Finding Relationships
elements.append(Paragraph("3. Finding Relationships", heading1_style))
elements.append(Paragraph("""
Further analysis explored the relationships between specific variables and crop yield.
""", normal_style))

# 3.1 Impact of Agricultural Inputs on Yield
elements.append(Paragraph("3.1 Impact of Agricultural Inputs on Yield", heading2_style))

for feature in ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]:
    if os.path.exists(os.path.join(plots_dir, f"{feature}_vs_yield.png")):
        elements.append(Image(os.path.join(plots_dir, f"{feature}_vs_yield.png"), width=6*inch, height=4*inch))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(f"Figure: Relationship between {feature} and Yield", caption_style))
        elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph("""
<b>Observations on Agricultural Inputs:</b> The scatter plots examining relationships between agricultural inputs and 
yield reveal several important patterns:

<b>Area vs. Yield:</b> There is no clear linear relationship between farm area and yield. This suggests that farm size 
alone does not determine productivity, and efficient farming practices can be implemented regardless of farm scale.

<b>Annual Rainfall vs. Yield:</b> While there is a slight positive trend, the relationship is not strong. This indicates 
that while adequate rainfall is necessary, excessive rainfall may not proportionally increase yields and may even be 
detrimental for certain crops.

<b>Fertilizer vs. Yield:</b> Despite expectations, there is no strong positive correlation between fertilizer application 
and yield. This counter-intuitive finding suggests that optimal fertilizer usage depends on specific crop requirements, 
soil conditions, and application methods rather than simply the quantity applied.

<b>Pesticide vs. Yield:</b> Similar to fertilizer, pesticide usage shows a weak relationship with yield. This may indicate 
that targeted pest management strategies could be more effective than broad application, or that over-application might 
even harm beneficial organisms that contribute to crop health.
""", normal_style))

# 3.2 Crop Yield by Season
elements.append(Paragraph("3.2 Crop Yield by Season", heading2_style))
if os.path.exists(os.path.join(plots_dir, "yield_by_crop_season.png")):
    elements.append(Image(os.path.join(plots_dir, "yield_by_crop_season.png"), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure: Average yield by crop and season for top performers", caption_style))
    elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph("""
<b>Observations on Crop-Season Combinations:</b> The analysis of yield by crop and season reveals important patterns for 
maximizing productivity:

- Coconut grown throughout the Whole Year consistently shows the highest yields, significantly outperforming all other 
  crop-season combinations.
- Sugarcane performs best during the Kharif (monsoon) season, likely due to its high water requirements.
- Several crops show distinct seasonal preferences, with some performing significantly better in specific seasons.
- The "Whole Year" cultivation approach generally produces higher yields for crops that can be grown in this manner, 
  suggesting advantages to continuous cultivation when possible.

These findings highlight the importance of matching crop selection with appropriate growing seasons to maximize yield 
potential.
""", normal_style))

# 4. Clustering Analysis
elements.append(Paragraph("4. Clustering Analysis", heading1_style))
elements.append(Paragraph("""
Farms were clustered based on area, annual rainfall, fertilizer, and pesticide usage to identify groups with similar 
characteristics and understand patterns in farming practices across India.
""", normal_style))

# 4.1 Determining Optimal Number of Clusters
elements.append(Paragraph("4.1 Determining Optimal Number of Clusters", heading2_style))
if os.path.exists(os.path.join(plots_dir, "cluster_evaluation.png")):
    elements.append(Image(os.path.join(plots_dir, "cluster_evaluation.png"), width=6*inch, height=3*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure: Elbow method and silhouette scores for determining the optimal number of clusters", caption_style))
    elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph(f"""
<b>Observations on Cluster Evaluation:</b> The cluster evaluation metrics guided the selection of the optimal number of 
clusters for this analysis. The elbow method plot shows the reduction in inertia (within-cluster sum of squares) as the 
number of clusters increases. The silhouette score, which measures how well samples fit within their assigned clusters, 
peaks at {summary['clustering_analysis']['optimal_clusters']} clusters, indicating this is the optimal number for 
segmenting the farms in our dataset. This balance provides meaningful differentiation between farm types while avoiding 
over-segmentation.
""", normal_style))

# 4.2 Cluster Visualization
elements.append(Paragraph("4.2 Cluster Visualization", heading2_style))
if os.path.exists(os.path.join(plots_dir, "cluster_pca.png")):
    elements.append(Image(os.path.join(plots_dir, "cluster_pca.png"), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure: Visualization of clusters using Principal Component Analysis", caption_style))
    elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph("""
<b>Observations on Cluster PCA:</b> The Principal Component Analysis (PCA) visualization reduces the dimensionality of our 
clustering features to two components, allowing us to visualize the farm segments. The distinct clusters are visible, with 
Cluster 3 (in green) being the largest and most dominant group, accounting for over 80% of all farms. The smaller clusters 
are more specialized farm types with distinct characteristics. The PCA plot shows some overlap between clusters, indicating 
gradual transitions between different farming systems rather than completely discrete categories.
""", normal_style))

# 4.3 Cluster Characteristics
elements.append(Paragraph("4.3 Cluster Characteristics", heading2_style))
if os.path.exists(os.path.join(plots_dir, "yield_by_cluster.png")):
    elements.append(Image(os.path.join(plots_dir, "yield_by_cluster.png"), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure: Yield distribution across different clusters", caption_style))
    elements.append(Spacer(1, 0.2*inch))

if os.path.exists(os.path.join(plots_dir, "features_by_cluster.png")):
    elements.append(Image(os.path.join(plots_dir, "features_by_cluster.png"), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Figure: Distribution of key features across clusters", caption_style))
    elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph(f"""
<b>Observations on Cluster Characteristics:</b> The analysis identified {summary['clustering_analysis']['optimal_clusters']} 
distinct farm clusters with the following key characteristics:

<b>Cluster 0 ({summary['clustering_analysis']['cluster_sizes']['0']} farms, 3.1%):</b> These are very large farms with 
moderate rainfall and extremely high fertilizer usage. With an average yield of just {summary['clustering_analysis']['cluster_yields']['0']:.2f}, 
these farms appear to be inefficient despite high resource inputs. Sugarcane, Wheat, and Maize are the predominant crops.

<b>Cluster 1 ({summary['clustering_analysis']['cluster_sizes']['1']} farms, 16.3%):</b> This cluster represents medium-sized 
farms with high rainfall and lower fertilizer usage. They achieve impressive average yields of {summary['clustering_analysis']['cluster_yields']['1']:.2f}, 
primarily growing Coconut, Sugarcane, and Tapioca. These farms appear to be leveraging natural rainfall effectively to 
minimize fertilizer inputs while maintaining high productivity.

<b>Cluster 2 ({summary['clustering_analysis']['cluster_sizes']['2']} farm, 0.005%):</b> This outlier cluster contains just one 
extremely large farm with high rainfall and extraordinarily high fertilizer usage. Despite the massive inputs, it achieves a low 
yield of {summary['clustering_analysis']['cluster_yields']['2']:.2f}, growing Niger seed. This unusual case may represent 
experimental farming, data error, or very specific conditions.

<b>Cluster 3 ({summary['clustering_analysis']['cluster_sizes']['3']} farms, 80.2%):</b> The dominant cluster consists of 
smaller farms with moderate rainfall and fertilizer usage. They achieve good average yields of {summary['clustering_analysis']['cluster_yields']['3']:.2f}, 
primarily growing Coconut, Sugarcane, and Banana. This cluster represents the typical farming system in India.

<b>Cluster 4 ({summary['clustering_analysis']['cluster_sizes']['4']} farms, 0.5%):</b> These are large farms with low rainfall and 
high fertilizer usage. They achieve relatively low yields of {summary['clustering_analysis']['cluster_yields']['4']:.2f}, primarily 
growing Wheat, Rice, and Cotton. These farms may be struggling with water scarcity despite trying to compensate with higher 
fertilizer application.

The clustering analysis reveals that farm size and input intensity do not necessarily correlate with higher yields. The most 
successful farms (Cluster 1) achieve high yields with moderate inputs and favorable natural conditions.
""", normal_style))

# 5. Recommendations
elements.append(Paragraph("5. Recommendations", heading1_style))

elements.append(Paragraph("5.1 Crop Selection", heading2_style))
elements.append(Paragraph(f"""
<b>High-yielding crops:</b> Coconut ({summary['top_crops']['by_yield']['Coconut ']:.2f}), Sugarcane ({summary['top_crops']['by_yield']['Sugarcane']:.2f}), 
and Banana ({summary['top_crops']['by_yield']['Banana']:.2f}) consistently show the highest yields across India and should be prioritized 
where growing conditions permit.

<b>Seasonal considerations:</b>
• Coconut performs best when grown throughout the whole year
• Sugarcane shows best results during the Kharif (monsoon) season
• Banana yields are highest during the Summer season

<b>Regional preferences:</b>
• Coconut yields are highest in Telangana
• Sugarcane performs well in Puducherry
• Banana shows exceptional yields in Gujarat
""", normal_style))

elements.append(Paragraph("5.2 Optimal Growing Conditions", heading2_style))
elements.append(Paragraph("""
<b>Coconut:</b> Best with moderate rainfall (around 746 mm annually)
<b>Sugarcane:</b> Performs well with higher rainfall (around 1,330 mm annually)
<b>Banana:</b> Thrives with moderate rainfall (around 1,220 mm annually)
""", normal_style))

elements.append(Paragraph("5.3 Resource Optimization", heading2_style))
elements.append(Paragraph("""
<b>Fertilizer usage:</b> Optimize based on crop type. High-yield crops like Coconut require moderate fertilizer application, 
while some crops may need more intensive fertilization. The weak correlation between fertilizer amount and yield suggests that 
application method and timing may be more important than quantity.

<b>Rainfall considerations:</b> Choose water-intensive crops in high-rainfall areas and drought-resistant varieties in 
low-rainfall regions. The analysis shows that crops have specific rainfall requirements for optimal performance.

<b>Land utilization:</b> Farm size appears to have minimal correlation with yield, suggesting that efficient farming practices 
may be more important than farm size. Small-scale farmers can achieve high yields with proper crop selection and resource 
management.
""", normal_style))

elements.append(Paragraph("5.4 Cluster-specific Recommendations", heading2_style))
elements.append(Paragraph("""
<b>For farms in Cluster 1 (high rainfall, lower fertilizer usage):</b> Ideal for water-intensive crops like Coconut and 
Sugarcane. Consider increasing fertilizer usage for potentially higher yields, but maintain the current emphasis on 
leveraging natural rainfall.

<b>For farms in Cluster 3 (moderate rainfall, smaller areas):</b> Focus on high-value crops like Coconut, Sugarcane, and 
Banana that perform well in these conditions. This typical farming system in India shows good balance between inputs and 
outputs.

<b>For farms in Cluster 0 and 4 (lower rainfall, larger areas):</b> Consider drought-resistant crops and optimize fertilizer 
usage for cost efficiency. These farms should focus on water conservation techniques and may need to reconsider their crop mix 
to improve yields.
""", normal_style))

# 6. Limitations of Analysis
elements.append(Paragraph("6. Limitations of Analysis", heading1_style))
elements.append(Paragraph("""
1. This analysis does not account for soil quality or type, which can significantly affect crop yields.
2. Weather variations beyond annual rainfall (such as temperature, humidity, and seasonal distribution of rainfall) are not considered.
3. Market forces affecting crop selection and economic viability are not included in the analysis.
4. The analysis assumes current agricultural practices and technologies and does not account for adoption of new farming methods.
5. The clustering analysis may be influenced by outliers, particularly in clusters with very few farms.
6. The data spans multiple years but does not specifically analyze year-over-year trends or the impact of climate change.
""", normal_style))

# Build the PDF
doc.build(elements)

print(f"PDF report generated: {report_path}") 