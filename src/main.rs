use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;

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

/// Load CSV data and parse it into a vector of vectors
fn load_csv(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
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

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "winequality-white.csv";
    let data = load_csv(file_path)?;

    let k = 3; // Number of clusters
    let max_iterations = 100; // Max iterations for k-means

    let (centroids, assignments) = k_means(&data, k, max_iterations);

    println!("Centroids:");
    for (i, centroid) in centroids.iter().enumerate() {
        println!("Cluster {}: {:?}", i, centroid);
    }

    println!("\nAssignments:");
    for (i, cluster) in assignments.iter().enumerate() {
        println!("Sample {}: Cluster {}", i, cluster);
    }

    Ok(())
}
