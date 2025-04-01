use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingData {
    pub id: usize,
    pub query: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub timestamp: u64,
}