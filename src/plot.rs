use plotters::prelude::*;

/// Plots clustered data points and centroids on a 2D graph.
pub fn plot_clusters(
    data: &Vec<Vec<f64>>,
    assignments: &Vec<usize>,
    centroids: &Vec<Vec<f64>>,
    x_index: usize,
    y_index: usize,
    file_name: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            data.iter().map(|row| row[x_index]).fold(f64::INFINITY, f64::min)
                ..data.iter().map(|row| row[x_index]).fold(f64::NEG_INFINITY, f64::max),
            data.iter().map(|row| row[y_index]).fold(f64::INFINITY, f64::min)
                ..data.iter().map(|row| row[y_index]).fold(f64::NEG_INFINITY, f64::max),
        )?;

    chart.configure_mesh().draw()?;

    let colors = generate_colors(centroids.len());
    for (i, point) in data.iter().enumerate() {
        let cluster = assignments[i];
        chart.draw_series(PointSeries::of_element(
            vec![(point[x_index], point[y_index])],
            3,
            colors[cluster].filled(),
            &|coord, size, color| EmptyElement::at(coord) + Circle::new((0, 0), size, color.filled()),
        ))?;
    }

    for centroid in centroids {
        chart.draw_series(PointSeries::of_element(
            vec![(centroid[x_index], centroid[y_index])],
            8,
            BLACK.filled(),
            &|coord, size, color| EmptyElement::at(coord) + Circle::new((0, 0), size, color.filled()),
        ))?;
    }
    Ok(())
}

/// Plots the Elbow Method graph to find the optimal number of clusters.
pub fn plot_elbow_method(wcss: &Vec<f64>, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let k_values: Vec<usize> = (1..=wcss.len()).collect();
    let mut chart = ChartBuilder::on(&root)
        .caption("Elbow Method", ("sans-serif", 30))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(1..wcss.len() + 1, 0.0..*wcss.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())?;

    chart.configure_mesh().x_desc("Number of Clusters").y_desc("WCSS").draw()?;
    chart.draw_series(LineSeries::new(k_values.iter().zip(wcss.iter()).map(|(k, w)| (*k, *w)), &RED))?;
    Ok(())
}

/// Generates a vector of colors based on the number of clusters.
fn generate_colors(k: usize) -> Vec<RGBColor> {
    let base_colors = vec![RED, BLUE, GREEN, MAGENTA, CYAN, YELLOW];
    (0..k).map(|i| base_colors[i % base_colors.len()].clone()).collect()
}
