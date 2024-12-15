use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use plotters::prelude::*;

/// Calculate the Euclidean distance between two points
fn euclidean_distance(p1: &[f64], p2: &[f64]) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// K-Means clustering algorithm
fn k_means(data: &[Vec<f64>], k: usize, max_iterations: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let n_samples = data.len();
    let n_features = data[0].len();

    // Initialize centroids randomly
    let mut centroids: Vec<Vec<f64>> = {
        let mut rng = thread_rng();
        data.choose_multiple(&mut rng, k)
            .cloned()
            .collect()
    };

    let mut assignments = vec![0; n_samples];

    for _ in 0..max_iterations {
        // Assign samples to the nearest centroid
        for (i, sample) in data.iter().enumerate() {
            assignments[i] = centroids
                .iter()
                .enumerate()
                .map(|(j, centroid)| (j, euclidean_distance(sample, centroid)))
                .min_by(|(_, dist_a), (_, dist_b)| dist_a.partial_cmp(dist_b).unwrap())
                .unwrap()
                .0;
        }

        // Recompute centroids
        let mut new_centroids = vec![vec![0.0; n_features]; k];
        let mut counts = vec![0; k];

        for (sample, &cluster) in data.iter().zip(assignments.iter()) {
            for (f, &value) in sample.iter().enumerate() {
                new_centroids[cluster][f] += value;
            }
            counts[cluster] += 1;
        }

        for (centroid, &count) in new_centroids.iter_mut().zip(counts.iter()) {
            if count > 0 {
                for value in centroid.iter_mut() {
                    *value /= count as f64;
                }
            }
        }

        if centroids == new_centroids {
            break;
        }

        centroids = new_centroids;
    }

    (centroids, assignments)
}

/// Calculate the within-cluster sum of squares (WCSS)
fn calculate_wcss(data: &Vec<Vec<f64>>, centroids: &Vec<Vec<f64>>, assignments: &Vec<usize>) -> f64 {
    let mut wcss = 0.0;

    for (i, point) in data.iter().enumerate() {
        let cluster = assignments[i];
        wcss += euclidean_distance(point, &centroids[cluster]).powi(2);
    }

    wcss
}

/// Find the optimal number of clusters using the Elbow Method
fn find_optimal_k(data: &Vec<Vec<f64>>, max_k: usize, max_iterations: usize) -> Vec<(usize, f64)> {
    let mut results = Vec::new();

    for k in 1..=max_k {
        let (centroids, assignments) = k_means(data, k, max_iterations);
        let wcss = calculate_wcss(data, &centroids, &assignments);
        results.push((k, wcss));
    }

    results
}

/// Plot the Elbow Method results
fn plot_elbow_method(results: &Vec<(usize, f64)>) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("elbow_method.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Elbow Method", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(1..results.len(), 0.0..results.iter().map(|(_, wcss)| *wcss).fold(0.0, f64::max))?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        LineSeries::new(
            results.iter().map(|(k, wcss)| (*k, *wcss)),
            &RED,
        )
    )?
    .label("WCSS")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    println!("Elbow Method plot saved as 'elbow_method.png'.");
    Ok(())
}

fn plot_clusters(
    data: &Vec<Vec<f64>>,
    assignments: &Vec<usize>,
    centroids: &Vec<Vec<f64>>,
    x_index: usize,
    y_index: usize,
    file_name: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_values: Vec<f64> = data.iter().map(|row| row[x_index]).collect();
    let y_values: Vec<f64> = data.iter().map(|row| row[y_index]).collect();
    let x_min = x_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Cluster Visualization", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    let colors = vec![&RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &YELLOW];

    // Plot data points
    for (i, point) in data.iter().enumerate() {
        let cluster = assignments[i];
        chart.draw_series(PointSeries::of_element(
            vec![(point[x_index], point[y_index])],
            3,
            colors[cluster % colors.len()],
            &|coord, size, color| EmptyElement::at(coord) + Circle::new((0, 0), size, color.filled()),
        ))?;
    }

    // Plot centroids with reduced size
    for centroid in centroids.iter() {
        chart.draw_series(PointSeries::of_element(
            vec![(centroid[x_index], centroid[y_index])],
            3, // Reduced size for centroids
            &BLACK,
            &|coord, size, color| EmptyElement::at(coord) + Circle::new((0, 0), size, color.filled()),
        ))?;
    }

    println!("Cluster plot saved as '{}'.", file_name);
    Ok(())
}


/// Load CSV data and parse it into a vector of vectors
fn load_csv(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b';')
        .from_path(file_path)?;

    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row = record
            .iter()
            .map(|x| x.parse::<f64>().unwrap_or(0.0))
            .collect::<Vec<f64>>();
        data.push(row);
    }
    Ok(data)
}

fn normalize(data: &mut Vec<Vec<f64>>) {
    let n_features = data[0].len();
    for j in 0..n_features {
        let min_val = data.iter().map(|row| row[j]).fold(f64::INFINITY, f64::min);
        let max_val = data.iter().map(|row| row[j]).fold(f64::NEG_INFINITY, f64::max);
        for row in data.iter_mut() {
            row[j] = (row[j] - min_val) / (max_val - min_val);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "winequality-white.csv";

    println!("Loading data...");
    let mut data = load_csv(file_path)?;
    println!("Normalizing data...");
    normalize(&mut data);
       

    let max_k = 10;
    let max_iterations = 100;

    println!("Finding optimal number of clusters using the Elbow Method...");
    let results = find_optimal_k(&data, max_k, max_iterations);
    plot_elbow_method(&results)?;

    let k = 3; // Adjust as needed
    println!("Running K-means clustering with k = {}...", k);
    let (centroids, assignments) = k_means(&data, k, max_iterations);

    plot_clusters(&data, &assignments, &centroids, 0, 1, "clusters.png")?;

    Ok(())
}
