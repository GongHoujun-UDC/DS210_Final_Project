use std::error::Error;
use csv::ReaderBuilder;

/// Loads a CSV file and returns its data as a Vec<Vec<f64>>.
pub fn load_csv(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().from_path(file_path)?;
    let mut data = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row = record.iter().map(|field| field.parse::<f64>().unwrap()).collect();
        data.push(row);
    }
    Ok(data)
}

/// Filters specific columns from the dataset based on their indices.
pub fn filter_columns(data: &Vec<Vec<f64>>, columns: &Vec<usize>) -> Vec<Vec<f64>> {
    data.iter()
        .map(|row| columns.iter().map(|&i| row[i]).collect())
        .collect()
}

/// Standardizes the dataset so that each feature has a mean of 0 and a standard deviation of 1.
pub fn standardize(data: &mut Vec<Vec<f64>>) {
    let n_features = data[0].len();
    let n_samples = data.len();

    for j in 0..n_features {
        let mean: f64 = data.iter().map(|row| row[j]).sum::<f64>() / n_samples as f64;
        let std_dev: f64 = (data.iter()
            .map(|row| (row[j] - mean).powi(2))
            .sum::<f64>() / n_samples as f64)
            .sqrt();

        for i in 0..n_samples {
            data[i][j] = (data[i][j] - mean) / std_dev;
        }
    }
}

/// Calculates the Euclidean distance between two points.
pub fn euclidean_distance(p1: &Vec<f64>, p2: &Vec<f64>) -> f64 {
    p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
}
