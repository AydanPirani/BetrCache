use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingData {
    pub id: usize,
    pub query: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub timestamp: u64,
}

impl EmbeddingData {
    pub fn new(id: usize, query: String, embedding: Vec<f32>, response: String, timestamp: u64) -> EmbeddingData {
        return EmbeddingData {
            id,
            query,
            embedding,
            response,
            timestamp,
        } 
    }
}
