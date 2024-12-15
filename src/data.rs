use csv::ReaderBuilder;
use std::error::Error;

/// Load CSV file into a Vec<Vec<f64>>
pub fn load_csv(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b';')
        .from_path(file_path)?;

    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        data.push(record.iter()
            .map(|x| x.parse::<f64>().unwrap_or(0.0))
            .collect());
    }
    Ok(data)
}

/// Filter specific columns from the dataset
pub fn filter_columns(data: &[Vec<f64>], selected_indices: &[usize]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|row| selected_indices.iter().map(|&i| row[i]).collect())
        .collect()
}

/// Standardize the dataset: mean = 0, standard deviation = 1
pub fn standardize(data: &mut Vec<Vec<f64>>) {
    let n_features = data[0].len();
    for j in 0..n_features {
        let mean = data.iter().map(|row| row[j]).sum::<f64>() / data.len() as f64;
        let std_dev = (data.iter()
            .map(|row| (row[j] - mean).powi(2))
            .sum::<f64>()
            / data.len() as f64)
            .sqrt();
        for row in data.iter_mut() {
            row[j] = (row[j] - mean) / std_dev;
        }
    }
}

/// Calculate the Euclidean distance between two points
pub fn euclidean_distance(p1: &[f64], p2: &[f64]) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}
