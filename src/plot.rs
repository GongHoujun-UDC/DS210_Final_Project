use plotters::prelude::*;

pub fn plot_clusters(
    data: &Vec<Vec<f64>>,
    assignments: &Vec<usize>,
    centroids: &Vec<Vec<f64>>,
    x_index: usize,
    y_index: usize,
    file_name: &str,
    title: &str, // Title for the graph
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Build chart with default size and grid
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30)) // Graph title
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            data.iter().map(|row| row[x_index]).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                ..data.iter().map(|row| row[x_index]).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            data.iter().map(|row| row[y_index]).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                ..data.iter().map(|row| row[y_index]).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        )?;

    chart.configure_mesh().draw()?; // Automatically draw grid

    // Generate dynamic colors for clusters
    let colors = generate_colors(centroids.len());

    // Plot data points
    for (i, point) in data.iter().enumerate() {
        let cluster = assignments[i];
        chart.draw_series(PointSeries::of_element(
            vec![(point[x_index], point[y_index])],
            3, // Data point size
            colors[cluster].filled(),
            &|coord, size, color| EmptyElement::at(coord) + Circle::new((0, 0), size, color.filled()),
        ))?;
    }

    // Plot centroids as black circles
    for centroid in centroids {
        chart.draw_series(PointSeries::of_element(
            vec![(centroid[x_index], centroid[y_index])],
            5, // Larger size for centroids
            BLACK.filled(),
            &|coord, size, color| EmptyElement::at(coord) + Circle::new((0, 0), size, color.filled()),
        ))?;
    }

    Ok(())
}

/// Generate a vector of colors based on the number of centroids
fn generate_colors(k: usize) -> Vec<RGBColor> {
    let base_colors = vec![
        RED, BLUE, GREEN, MAGENTA, CYAN, YELLOW];
    let mut colors = Vec::new();
    for i in 0..k {
        colors.push(base_colors[i % base_colors.len()].clone());
    }
    colors
}

pub fn plot_elbow_method(wcss: &Vec<f64>, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let k_values: Vec<usize> = (1..=wcss.len()).collect();

    let max_wcss = *wcss.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Elbow Method", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(1..wcss.len() + 1, 0.0..max_wcss)?;

    chart.configure_mesh()
        .x_desc("Number of Clusters (k)")
        .y_desc("WCSS")
        .draw()?;

    chart.draw_series(LineSeries::new(
        k_values.iter().zip(wcss.iter()).map(|(k, w)| (*k, *w)),
        &RED,
    ))?
    .label("WCSS")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().draw()?;
    Ok(())
}