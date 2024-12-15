mod kmeans;
mod plot;
mod data;

use crate::data::{load_csv, filter_columns, standardize, euclidean_distance};
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
        let (centroids, assignments) = k_means(&filtered_data, k, 50);
        let total_wcss: f64 = filtered_data
            .iter()
            .zip(assignments.iter())
            .map(|(point, &cluster)| euclidean_distance(point, &centroids[cluster]))
            .sum();
        wcss.push(total_wcss);
    }
    plot_elbow_method(&wcss, "elbow_method.png")?;
    println!("Elbow Method plot saved as 'elbow_method.png'.");

    let k = 3;
    println!("Running K-means clustering with k = {}...", k);
    let (centroids, assignments) = k_means(&filtered_data, k, 50);

    println!("Plotting 'density' vs 'residual sugar'...");
    plot_clusters(
        &filtered_data,
        &assignments,
        &centroids,
        0,
        1,
        "clusters_density_sugar.png",
        "Density vs Residual Sugar",
    )?;
    println!("Cluster plot saved as 'clusters_density_sugar.png'.");

    println!("Plotting 'alcohol' vs 'chlorides'...");
    plot_clusters(
        &filtered_data,
        &assignments,
        &centroids,
        2,
        3,
        "clusters_alcohol_chlorides.png",
        "Alcohol vs Chlorides",
    )?;
    println!("Cluster plot saved as 'clusters_alcohol_chlorides.png'.");

    println!("Clustering completed!");
    Ok(())
}
