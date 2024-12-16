mod data;
mod kmeans;
mod plot;

use crate::data::{load_csv, filter_columns, standardize};
use crate::kmeans::k_means;
use crate::plot::{plot_clusters, plot_elbow_method};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "winequality-white.csv";
    let mut data = load_csv(file_path)?;
    standardize(&mut data);

    let selected_columns = vec![7, 3, 10, 4];
    let filtered_data = filter_columns(&data, &selected_columns);

    let mut wcss = Vec::new();
    for k in 1..=10 {
        let (centroids, assignments) = k_means(&filtered_data, k, 100);
        let total_wcss: f64 = filtered_data.iter().zip(assignments.iter())
            .map(|(point, &cluster)| crate::data::euclidean_distance(point, &centroids[cluster]))
            .sum();
        wcss.push(total_wcss);
    }
    plot_elbow_method(&wcss, "elbow_method.png")?;

    let k = 3;
    let (centroids, assignments) = k_means(&filtered_data, k, 100);
    plot_clusters(&filtered_data, &assignments, &centroids, 0, 1, "clusters_density_sugar.png", "Density vs Residual Sugar")?;
    plot_clusters(&filtered_data, &assignments, &centroids, 2, 3, "clusters_alcohol_chlorides.png", "Alcohol vs Chlorides")?;

    Ok(())
}




#[cfg(test)]
mod tests {
    use crate::data::{filter_columns, standardize, euclidean_distance};
    use crate::kmeans::{k_means, initialize_centroids};
    use crate::plot::{plot_clusters, plot_elbow_method};
    use std::fs;

    // Test: Filter specific columns
    #[test]
    fn test_filter_columns() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let selected_columns = vec![0, 2];
        let filtered = filter_columns(&data, &selected_columns);

        assert_eq!(filtered.len(), 2, "Row count should match");
        assert_eq!(filtered[0], vec![1.0, 3.0], "Filtered row content should match");
    }

    // Test: Standardize the dataset
    #[test]
    fn test_standardize() {
        let mut data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        standardize(&mut data);

        let mean_0: f64 = data.iter().map(|row| row[0]).sum::<f64>() / data.len() as f64;
        let mean_1: f64 = data.iter().map(|row| row[1]).sum::<f64>() / data.len() as f64;

        assert!(mean_0.abs() < 1e-6, "Column 0 mean should be close to 0");
        assert!(mean_1.abs() < 1e-6, "Column 1 mean should be close to 0");
    }

    // Test: Euclidean distance calculation
    #[test]
    fn test_euclidean_distance() {
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![4.0, 6.0, 3.0];
        let distance = euclidean_distance(&p1, &p2);

        assert_eq!(distance, 5.0, "Euclidean distance should be correct");
    }

    // Test: Initialize centroids
    #[test]
    fn test_initialize_centroids() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let centroids = initialize_centroids(&data, 2);
        assert_eq!(centroids.len(), 2, "Should initialize 2 centroids");
    }

    // Test: K-means clustering
    #[test]
    fn test_k_means() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![5.0, 6.0],
            vec![5.1, 6.1],
        ];
        let k = 2;
        let (centroids, assignments) = k_means(&data, k, 100);

        assert_eq!(centroids.len(), k, "Centroid count should match k");
        assert_eq!(assignments.len(), data.len(), "All points should be assigned to clusters");
    }

    // Test: Plot clusters (output file creation)
    #[test]
    fn test_plot_clusters() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let assignments = vec![0, 1, 0];
        let centroids = vec![vec![2.0, 3.0], vec![4.0, 5.0]];

        let result = plot_clusters(&data, &assignments, &centroids, 0, 1, "test_clusters.png", "Test Clusters");
        assert!(result.is_ok(), "Plotting should succeed");
        assert!(fs::metadata("test_clusters.png").is_ok(), "Output file should exist");
        fs::remove_file("test_clusters.png").unwrap();
    }

    // Test: Plot elbow method
    #[test]
    fn test_plot_elbow_method() {
        let wcss = vec![1000.0, 800.0, 600.0, 400.0, 300.0];
        let result = plot_elbow_method(&wcss, "test_elbow.png");

        assert!(result.is_ok(), "Elbow plot should succeed");
        assert!(fs::metadata("test_elbow.png").is_ok(), "Elbow output file should exist");
        fs::remove_file("test_elbow.png").unwrap();
    }
}
