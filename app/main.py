#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop Yield Analysis in India

This script analyzes crop yield data from India to explore relationships between
various factors and crop yields, identify farm clusters, and provide recommendations
for farmers to improve their yields.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
import json
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/crop_yield_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create output directory for plots
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Create output directory for data
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to plots directory"""
    fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_dataframe(df, filename):
    """Save dataframe to output directory as CSV"""
    df.to_csv(os.path.join(output_dir, filename), index=True)
    logger.info(f"Saved dataframe to {os.path.join(output_dir, filename)}")

def log_dataframe(df, description, max_rows=10):
    """Log dataframe information and first few rows"""
    logger.info(f"\n{description}:\n")
    logger.info(f"\nShape: {df.shape}")
    
    # Convert to string representation
    if len(df) <= max_rows:
        df_str = df.to_string()
    else:
        df_str = df.head(max_rows).to_string() + f"\n... ({len(df) - max_rows} more rows)"
    
    logger.info(f"\n{df_str}\n")

def main():
    """Main analysis function"""
    logger.info("Starting Crop Yield Analysis")
    
    # Load the data
    try:
        logger.info("Loading dataset")
        data = pd.read_csv('docs/crop_yield.csv')
        logger.info(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return
    
    #---------------------------------------------------------------------------
    # 1. Data Exploration
    #---------------------------------------------------------------------------
    logger.info("\n" + "="*80 + "\n" + "DATA EXPLORATION" + "\n" + "="*80)
    
    # Display the first few rows
    log_dataframe(data.head(), "First 5 rows of the dataset")
    
    # Display information about the dataset
    buffer = []
    buffer.append("Dataset Information:")
    buffer.append(f"Total Records: {data.shape[0]}")
    buffer.append(f"Total Features: {data.shape[1]}")
    buffer.append("\nData Types:")
    for col, dtype in data.dtypes.items():
        buffer.append(f"  - {col}: {dtype}")
    
    buffer.append("\nMissing Values:")
    for col, count in data.isnull().sum().items():
        buffer.append(f"  - {col}: {count}")
    
    logger.info("\n".join(buffer))
    
    # Calculate basic statistics for numerical columns
    numeric_data = data.select_dtypes(include=[np.number])
    log_dataframe(numeric_data.describe(), "Basic Statistics for Numerical Features")
    save_dataframe(numeric_data.describe(), "numerical_statistics.csv")
    
    # Count unique values in categorical columns
    categorical_cols = ['Crop', 'Season', 'State']
    buffer = ["Unique Values in Categorical Columns:"]
    for col in categorical_cols:
        buffer.append(f"  - {col}: {data[col].nunique()} unique values")
        
        # Log top values
        value_counts = data[col].value_counts().head(10)
        buffer.append(f"    Top 10 {col} values:")
        for val, count in value_counts.items():
            buffer.append(f"      - {val}: {count} ({count/len(data)*100:.2f}%)")
            
    logger.info("\n".join(buffer))
    
    #---------------------------------------------------------------------------
    # Visualizing the Data
    #---------------------------------------------------------------------------
    logger.info("\n" + "-"*80 + "\n" + "DATA VISUALIZATION" + "\n" + "-"*80)
    
    # Visualize distribution of yield
    logger.info("Generating yield distribution plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Yield'], kde=True, ax=ax)
    ax.set_title('Distribution of Crop Yield')
    ax.set_xlabel('Yield (tons per hectare)')
    ax.grid(True, alpha=0.3)
    
    # Log yield distribution statistics
    yield_stats = data['Yield'].describe()
    logger.info(f"Yield Distribution Statistics:\n{yield_stats.to_string()}")
    
    # Save the plot
    save_figure(fig, "yield_distribution.png")
    logger.info("Saved yield distribution plot")
    
    # Boxplot of yield by top crops
    logger.info("Generating boxplot of yield by top crops")
    top_10_crops = data['Crop'].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Crop', y='Yield', data=data[data['Crop'].isin(top_10_crops)], ax=ax)
    ax.set_title('Yield Distribution by Top 10 Crops')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Log crop yield statistics
    crop_yield_stats = data[data['Crop'].isin(top_10_crops)].groupby('Crop')['Yield'].describe()
    log_dataframe(crop_yield_stats, "Yield Statistics by Top 10 Crops")
    save_dataframe(crop_yield_stats, "top_crops_yield_stats.csv")
    
    # Save the plot
    save_figure(fig, "yield_by_top_crops.png")
    logger.info("Saved yield by top crops plot")
    
    # Boxplot of yield by season
    logger.info("Generating boxplot of yield by season")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Season', y='Yield', data=data, ax=ax)
    ax.set_title('Yield Distribution by Season')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Log season yield statistics
    season_yield_stats = data.groupby('Season')['Yield'].describe()
    log_dataframe(season_yield_stats, "Yield Statistics by Season")
    save_dataframe(season_yield_stats, "season_yield_stats.csv")
    
    # Save the plot
    save_figure(fig, "yield_by_season.png")
    logger.info("Saved yield by season plot")
    
    #---------------------------------------------------------------------------
    # Correlation Analysis
    #---------------------------------------------------------------------------
    logger.info("\n" + "-"*80 + "\n" + "CORRELATION ANALYSIS" + "\n" + "-"*80)
    
    # Calculate correlation matrix
    correlation_matrix = numeric_data.corr()
    log_dataframe(correlation_matrix, "Correlation Matrix of Numerical Features")
    save_dataframe(correlation_matrix, "correlation_matrix.csv")
    
    # Visualize correlation matrix as heatmap
    logger.info("Generating correlation heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, linecolor='black', ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features')
    
    # Save the plot
    save_figure(fig, "correlation_heatmap.png")
    logger.info("Saved correlation heatmap")
    
    # Extract correlations with yield
    yield_correlations = correlation_matrix['Yield'].sort_values(ascending=False)
    logger.info(f"Features Correlation with Yield:\n{yield_correlations.to_string()}")
    save_dataframe(yield_correlations.to_frame(name='Correlation'), "yield_correlations.csv")
    
    # Visualize the correlations with yield
    logger.info("Generating yield correlations bar plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    yield_correlations.drop('Yield').plot(kind='bar', ax=ax)
    ax.set_title('Correlation of Features with Yield')
    ax.set_ylabel('Correlation Coefficient')
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    save_figure(fig, "yield_correlations.png")
    logger.info("Saved yield correlations plot")
    
    #---------------------------------------------------------------------------
    # 2. Clustering Analysis
    #---------------------------------------------------------------------------
    logger.info("\n" + "="*80 + "\n" + "CLUSTERING ANALYSIS" + "\n" + "="*80)
    
    # Select features for clustering
    cluster_features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    X = data[cluster_features]
    
    # Standardize the features
    logger.info("Standardizing features for clustering")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using the Elbow Method
    logger.info("Determining optimal number of clusters")
    inertia = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        logger.info(f"Fitting K-means with {k} clusters")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        logger.info(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.4f}")
    
    # Save cluster evaluation metrics
    cluster_evaluation = pd.DataFrame({
        'K': list(K_range),
        'Inertia': inertia,
        'Silhouette_Score': silhouette_scores
    })
    save_dataframe(cluster_evaluation, "cluster_evaluation.csv")
    
    # Plot Elbow Method and Silhouette Scores
    logger.info("Generating elbow method and silhouette score plots")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(K_range, inertia, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal k')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "cluster_evaluation.png")
    logger.info("Saved cluster evaluation plots")
    
    # Choose optimal number of clusters based on evaluation
    # Here we're using the silhouette score to determine the optimal k
    # A higher silhouette score indicates better-defined clusters
    optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    logger.info(f"Optimal number of clusters determined to be: {optimal_k}")
    
    # Fit K-means with the optimal number of clusters
    logger.info(f"Fitting final K-means model with {optimal_k} clusters")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze cluster characteristics
    cluster_stats = data.groupby('Cluster')[cluster_features + ['Yield']].mean()
    log_dataframe(cluster_stats, "Cluster Characteristics")
    save_dataframe(cluster_stats, "cluster_characteristics.csv")
    
    # Also calculate standard deviations and sizes of clusters
    cluster_stds = data.groupby('Cluster')[cluster_features + ['Yield']].std()
    cluster_sizes = data.groupby('Cluster').size()
    
    cluster_info = pd.DataFrame({
        'Size': cluster_sizes,
        'Percentage': cluster_sizes / len(data) * 100
    })
    
    log_dataframe(cluster_info, "Cluster Sizes")
    save_dataframe(cluster_info, "cluster_sizes.csv")
    
    # Visualize clusters using PCA for dimensionality reduction
    logger.info("Performing PCA for cluster visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Log PCA explained variance
    explained_variance = pca.explained_variance_ratio_
    logger.info(f"PCA explained variance: {explained_variance[0]:.4f}, {explained_variance[1]:.4f}")
    logger.info(f"Total explained variance: {sum(explained_variance):.4f}")
    
    # Plot clusters
    logger.info("Generating PCA cluster visualization")
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    ax.set_title('Farm Clusters Visualized Using PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, "cluster_pca.png")
    logger.info("Saved PCA cluster visualization")
    
    # Box plots to compare yield across clusters
    logger.info("Generating yield distribution by cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Yield', data=data, ax=ax)
    ax.set_title('Yield Distribution by Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Yield (tons per hectare)')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, "yield_by_cluster.png")
    logger.info("Saved yield distribution by cluster")
    
    # Compare other features across clusters
    logger.info("Generating feature distributions by cluster")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(cluster_features):
        sns.boxplot(x='Cluster', y=feature, data=data, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution by Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "features_by_cluster.png")
    logger.info("Saved feature distributions by cluster")
    
    #---------------------------------------------------------------------------
    # 3. Finding Relationships
    #---------------------------------------------------------------------------
    logger.info("\n" + "="*80 + "\n" + "RELATIONSHIP ANALYSIS" + "\n" + "="*80)
    
    # Analyze the relationship between features and yield
    logger.info("Analyzing relationships between features and yield")
    for feature in ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']:
        logger.info(f"Analyzing relationship between {feature} and Yield")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=feature, y='Yield', data=data, scatter_kws={'alpha':0.3}, ax=ax)
        ax.set_title(f'Relationship Between {feature} and Yield')
        ax.set_xlabel(feature)
        ax.set_ylabel('Yield (tons per hectare)')
        ax.grid(True, alpha=0.3)
        
        # Calculate and log correlation
        correlation = data[[feature, 'Yield']].corr().iloc[0, 1]
        logger.info(f"Correlation between {feature} and Yield: {correlation:.4f}")
        
        save_figure(fig, f"{feature}_vs_yield.png")
        logger.info(f"Saved {feature} vs yield plot")
    
    # Analyze yield by crop and season
    logger.info("Analyzing yield by crop and season")
    crop_season_yield = data.groupby(['Crop', 'Season'])['Yield'].mean().reset_index()
    crop_season_yield = crop_season_yield.sort_values('Yield', ascending=False).head(15)
    
    log_dataframe(crop_season_yield, "Top 15 Crop-Season Combinations by Average Yield")
    save_dataframe(crop_season_yield, "top_crop_season_yield.csv")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Crop', y='Yield', hue='Season', data=crop_season_yield, ax=ax)
    ax.set_title('Average Yield by Crop and Season (Top 15)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Season')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "yield_by_crop_season.png")
    logger.info("Saved yield by crop and season plot")
    
    # Association Rule Mining
    logger.info("Performing Association Rule Mining")
    
    # First, we need to discretize the continuous variables
    logger.info("Discretizing continuous variables for association rule mining")
    data_for_rules = data.copy()
    
    # Discretize numerical columns
    for col in ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']:
        data_for_rules[col + '_bin'] = pd.qcut(data_for_rules[col], q=3, labels=['Low', 'Medium', 'High'])
    
    # Create binary columns for association rules
    logger.info("Creating binary columns for association rule mining")
    
    def encode_features(df):
        result = df.copy()
        for column in ['Crop', 'Season', 'State', 'Area_bin', 'Annual_Rainfall_bin', 
                      'Fertilizer_bin', 'Pesticide_bin', 'Yield_bin']:
            dummies = pd.get_dummies(df[column], prefix=column)
            result = pd.concat([result, dummies], axis=1)
        return result
    
    # Encode features
    data_encoded = encode_features(data_for_rules)
    
    # Select the binary columns
    binary_cols = [col for col in data_encoded.columns if '_' in col and col not in 
                  ['Crop_Year', 'Area_bin', 'Annual_Rainfall_bin', 'Fertilizer_bin', 
                   'Pesticide_bin', 'Yield_bin']]
    
    # Prepare the dataset for apriori algorithm
    basket = data_encoded[binary_cols].astype(bool)
    
    # Apply the Apriori algorithm
    logger.info("Applying Apriori algorithm with min_support=0.05")
    try:
        frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
        logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        logger.info("Generating association rules with min confidence=0.5")
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        logger.info(f"Generated {len(rules)} association rules")
        
        # Sort rules by lift
        rules = rules.sort_values('lift', ascending=False)
        
        # Save association rules
        save_dataframe(rules, "association_rules.csv")
        
        # Log top 20 rules
        log_dataframe(rules.head(20), "Top 20 Association Rules (by Lift)")
        
        # Visualize some of the top rules
        if len(rules) >= 50:
            logger.info("Generating visualization of top association rules")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x='support', y='confidence', size='lift', 
                           data=rules.head(50), sizes=(50, 500), alpha=0.7, ax=ax)
            ax.set_title('Support vs Confidence for Top 50 Rules')
            ax.grid(True, alpha=0.3)
            
            save_figure(fig, "association_rules.png")
            logger.info("Saved association rules visualization")
    except Exception as e:
        logger.error(f"Error in association rule mining: {str(e)}")
    
    #---------------------------------------------------------------------------
    # 4. Making Recommendations
    #---------------------------------------------------------------------------
    logger.info("\n" + "="*80 + "\n" + "RECOMMENDATIONS" + "\n" + "="*80)
    
    # Identify high-yield crops
    logger.info("Identifying high-yield crops")
    high_yield_crops = data.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(10)
    log_dataframe(high_yield_crops.to_frame(), "Top 10 Crops by Average Yield")
    save_dataframe(high_yield_crops.to_frame(), "high_yield_crops.csv")
    
    # Identify optimal conditions for top 5 crops
    top_5_crops = high_yield_crops.index[:5]
    crop_recommendations = []
    
    for crop in top_5_crops:
        logger.info(f"Analyzing optimal conditions for {crop}")
        crop_data = data[data['Crop'] == crop]
        
        if len(crop_data) == 0:
            logger.warning(f"No data found for crop: {crop}")
            continue
            
        rec = {"Crop": crop}
        
        # Best season
        best_season = crop_data.groupby('Season')['Yield'].mean().sort_values(ascending=False)
        if len(best_season) > 0:
            rec["Best_Season"] = best_season.index[0]
            rec["Season_Avg_Yield"] = best_season.iloc[0]
            logger.info(f"Best Season for {crop}: {best_season.index[0]} (Average Yield: {best_season.iloc[0]:.2f})")
        
        # Best state
        best_state = crop_data.groupby('State')['Yield'].mean().sort_values(ascending=False)
        if len(best_state) > 0:
            rec["Best_State"] = best_state.index[0]
            rec["State_Avg_Yield"] = best_state.iloc[0]
            logger.info(f"Best State for {crop}: {best_state.index[0]} (Average Yield: {best_state.iloc[0]:.2f})")
        
        # Optimal rainfall
        if len(crop_data) > 0:
            max_yield_idx = crop_data['Yield'].idxmax()
            optimal_rainfall = crop_data.loc[max_yield_idx, 'Annual_Rainfall']
            rec["Optimal_Rainfall"] = optimal_rainfall
            logger.info(f"Optimal Annual Rainfall for {crop}: {optimal_rainfall:.2f} mm")
            
            # Optimal fertilizer and pesticide usage
            optimal_fertilizer = crop_data.loc[max_yield_idx, 'Fertilizer']
            optimal_pesticide = crop_data.loc[max_yield_idx, 'Pesticide']
            rec["Optimal_Fertilizer"] = optimal_fertilizer
            rec["Optimal_Pesticide"] = optimal_pesticide
            logger.info(f"Optimal Fertilizer Usage for {crop}: {optimal_fertilizer:.2f} kg")
            logger.info(f"Optimal Pesticide Usage for {crop}: {optimal_pesticide:.2f} kg")
        
        crop_recommendations.append(rec)
    
    # Save crop recommendations
    crop_recommendations_df = pd.DataFrame(crop_recommendations)
    save_dataframe(crop_recommendations_df, "crop_recommendations.csv")
    
    # Recommendations based on clusters
    logger.info("Generating recommendations based on farm clusters")
    cluster_recommendations = []
    
    for cluster in range(optimal_k):
        logger.info(f"Analyzing Cluster {cluster}")
        cluster_data = data[data['Cluster'] == cluster]
        avg_yield = cluster_data['Yield'].mean()
        
        rec = {
            "Cluster": cluster,
            "Average_Yield": avg_yield,
            "Size": len(cluster_data),
            "Percentage": len(cluster_data) / len(data) * 100
        }
        
        logger.info(f"Cluster {cluster} (Average Yield: {avg_yield:.2f}):")
        
        # Characteristics of this cluster
        logger.info("Characteristics:")
        for feature in cluster_features:
            avg_value = cluster_data[feature].mean()
            rec[f"Average_{feature}"] = avg_value
            logger.info(f"  - Average {feature}: {avg_value:.2f}")
        
        # Best crops for this cluster
        best_crops = cluster_data.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(3)
        
        logger.info("Recommended Crops:")
        for i, (crop, yield_value) in enumerate(best_crops.items()):
            rec[f"Recommended_Crop_{i+1}"] = crop
            rec[f"Expected_Yield_{i+1}"] = yield_value
            logger.info(f"  - {crop} (Expected Yield: {yield_value:.2f})")
        
        # Recommendations
        recommendations = []
        
        if cluster_data['Annual_Rainfall'].mean() > data['Annual_Rainfall'].mean():
            recommendations.append("This cluster has high rainfall. Consider water-intensive crops.")
            logger.info("  - This cluster has high rainfall. Consider water-intensive crops.")
        else:
            recommendations.append("This cluster has low rainfall. Consider drought-resistant crops.")
            logger.info("  - This cluster has low rainfall. Consider drought-resistant crops.")
            
        if cluster_data['Fertilizer'].mean() > data['Fertilizer'].mean():
            recommendations.append("Consider optimizing fertilizer usage for cost efficiency.")
            logger.info("  - Consider optimizing fertilizer usage for cost efficiency.")
        else:
            recommendations.append("Consider increasing fertilizer usage for potentially higher yields.")
            logger.info("  - Consider increasing fertilizer usage for potentially higher yields.")
        
        rec["Recommendations"] = "; ".join(recommendations)
        cluster_recommendations.append(rec)
    
    # Save cluster recommendations
    cluster_recommendations_df = pd.DataFrame(cluster_recommendations)
    save_dataframe(cluster_recommendations_df, "cluster_recommendations.csv")
    
    #---------------------------------------------------------------------------
    # 5. Summary of Findings
    #---------------------------------------------------------------------------
    logger.info("\n" + "="*80 + "\n" + "SUMMARY OF FINDINGS" + "\n" + "="*80)
    
    logger.info("Key Factors Affecting Crop Yield:")
    for feature, corr in yield_correlations.items():
        logger.info(f"  - {feature}: {corr:.4f}")
    
    logger.info("\nFarm Clusters:")
    for cluster in range(optimal_k):
        logger.info(f"  - Cluster {cluster}: {cluster_stats.loc[cluster, 'Yield']:.2f} average yield")
    
    logger.info("\nTop Performing Crops:")
    for crop, yield_value in high_yield_crops.items():
        logger.info(f"  - {crop}: {yield_value:.2f}")
    
    logger.info("\nRecommendations:")
    logger.info("1. Crop Selection:")
    logger.info("   - Choose crops suitable for your region's rainfall patterns.")
    logger.info("   - Consider seasonal variations in crop performance.")
    
    logger.info("\n2. Resource Optimization:")
    logger.info("   - Adjust fertilizer and pesticide usage based on crop requirements.")
    logger.info("   - Balance area under cultivation with resource availability.")
    
    logger.info("\n3. Cluster-Specific Recommendations:")
    for cluster in range(optimal_k):
        logger.info(f"   - Cluster {cluster}: {cluster_stats.loc[cluster, 'Yield']:.2f} average yield")
    
    logger.info("\nLimitations of Analysis:")
    logger.info("1. This analysis does not account for soil quality or type.")
    logger.info("2. Weather variations beyond annual rainfall are not considered.")
    logger.info("3. Market forces affecting crop selection are not included.")
    logger.info("4. The analysis assumes current agricultural practices and technologies.")
    
    #---------------------------------------------------------------------------
    # 6. Save Results for Report Generation
    #---------------------------------------------------------------------------
    logger.info("\n" + "="*80 + "\n" + "SAVING RESULTS" + "\n" + "="*80)
    
    # Save cluster assignments back to CSV for future reference
    data.to_csv(os.path.join(output_dir, 'crop_yield_with_clusters.csv'), index=False)
    logger.info("Saved crop yield data with cluster assignments")
    
    # Save a summary of all outputs for report generation
    summary = {
        "data_summary": {
            "total_records": len(data),
            "total_features": len(data.columns),
            "unique_crops": data['Crop'].nunique(),
            "unique_states": data['State'].nunique(),
            "unique_seasons": data['Season'].nunique(),
            "year_range": [int(data['Crop_Year'].min()), int(data['Crop_Year'].max())]
        },
        "correlation_analysis": {
            "yield_correlations": yield_correlations.to_dict(),
            "top_correlated_feature": yield_correlations.index[1],  # Index 0 is Yield itself
            "top_correlation_value": yield_correlations.iloc[1]
        },
        "clustering_analysis": {
            "optimal_clusters": optimal_k,
            "cluster_sizes": cluster_sizes.to_dict(),
            "cluster_yields": cluster_stats['Yield'].to_dict()
        },
        "top_crops": {
            "by_yield": high_yield_crops.to_dict(),
            "top_crop": high_yield_crops.index[0],
            "top_crop_yield": high_yield_crops.iloc[0]
        },
        "generated_files": {
            "plots": os.listdir(plots_dir),
            "data_outputs": os.listdir(output_dir)
        }
    }
    
    # Save summary as JSON
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info("Saved analysis summary for report generation")
    logger.info(f"Analysis complete! Results have been saved to {output_dir} and plots to {plots_dir}")
    logger.info(f"Log file created at {log_filename}")

if __name__ == "__main__":
    main()
