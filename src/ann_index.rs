use hnsw_rs::prelude::*;

pub type ANNResult<T> = Result<T, Box<dyn std::error::Error>>;

pub trait ANNIndex<'a> {
    fn init_index(&mut self, max_elements: usize, dimension: usize) -> ANNResult<()>;
    fn add_pt(&mut self, point: Vec<f32>, id: usize) -> ANNResult<()>;
    fn get_curr_ct(&self) -> ANNResult<(usize)>;
    fn get_max_elements(&mut self) -> ANNResult<(usize)>;
    fn resize(&mut self, new_size: usize) -> ANNResult<()>; 
    fn search_knn(&self, query: &[f32], k: usize) -> ANNResult<Vec<(usize, f32)>>;
}


pub struct HnswAnnIndex<'a> {
    hnsw: Hnsw<'a, f32, DistCosine>,
    dimension: usize,
    max_elements: usize,
    points: Vec<(Vec<f32>, usize)>,
}

impl<'a> ANNIndex<'a> for HnswAnnIndex<'a> {
    fn init_index(&mut self, max_elements: usize, dimension: usize) -> ANNResult<()> {
        self.hnsw = Hnsw::new(16, max_elements, dimension, 16, DistCosine);
        self.dimension = dimension;
        self.max_elements = max_elements;
        self.points = Vec::new();
        return Ok(());
    }

    fn add_pt(&mut self, point: Vec<f32>, id: usize) -> ANNResult<()> {
        if point.len() != self.dimension {
            return Err("Point dimensions don't match!".into());
        }

        self.hnsw.insert((&point, id));
        self.points.push((point, id));
        
        return Ok(());
    }

    fn get_curr_ct(&self) -> ANNResult<(usize)> {
        let ct = self.hnsw.get_nb_point();
        return Ok(ct);
    }

    fn get_max_elements(&mut self) -> ANNResult<(usize)> {
        return Ok(self.max_elements);
    }

    fn resize(&mut self, new_size: usize) -> ANNResult<()> {
        let hnsw = Hnsw::new(16, new_size, self.dimension, 16, DistCosine);
        
        // Reinsert each stored point into the new index.
        for (point, id) in &self.points {
            hnsw.insert((&point, *id));
        }

        self.hnsw = hnsw;
        self.max_elements = new_size;

        return Ok(());
    }

    fn search_knn(&self, query: &[f32], k: usize) -> ANNResult<Vec<(usize, f32)>> {
        let res = self.hnsw.search(query, k, k+2);

        let knn_results: Vec<(usize, f32)> = res.into_iter()
        .map(|result| (result.d_id, result.distance))
        .collect();

        return Ok(knn_results);
    }

    
}