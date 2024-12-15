use crate::data::euclidean_distance;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

pub fn k_means(data: &[Vec<f64>], k: usize, max_iterations: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let n_samples = data.len();
    let n_features = data[0].len();

    let mut centroids = initialize_centroids(data, k);
    let mut assignments = vec![0; n_samples];

    for _ in 0..max_iterations {
        // Assign to nearest centroids
        for (i, sample) in data.iter().enumerate() {
            assignments[i] = centroids
                .iter()
                .enumerate()
                .map(|(j, centroid)| (j, euclidean_distance(sample, centroid)))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
        }

        // Recompute centroids
        let mut new_centroids = vec![vec![0.0; n_features]; k];
        let mut counts = vec![0; k];
        for (sample, &cluster) in data.iter().zip(&assignments) {
            for (f, &value) in sample.iter().enumerate() {
                new_centroids[cluster][f] += value;
            }
            counts[cluster] += 1;
        }
        for (centroid, &count) in new_centroids.iter_mut().zip(&counts) {
            if count > 0 {
                for value in centroid.iter_mut() {
                    *value /= count as f64;
                }
            }
        }
        centroids = new_centroids;
    }

    (centroids, assignments)
}

pub fn initialize_centroids(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let mut centroids = Vec::new();
    centroids.push(data.choose(&mut rng).unwrap().clone());

    while centroids.len() < k {
        let mut distances: Vec<f64> = data.iter()
            .map(|point| centroids.iter()
                .map(|c| euclidean_distance(point, c))
                .fold(f64::INFINITY, f64::min))
            .collect();

        let total_distance: f64 = distances.iter().sum();
        distances.iter_mut().for_each(|d| *d /= total_distance);

        let rand_value: f64 = rng.gen();
        let mut cumulative = 0.0;
        let selected_index = distances.iter()
            .position(|&d| { cumulative += d; cumulative >= rand_value })
            .unwrap();
        centroids.push(data[selected_index].clone());
    }
    centroids
}
