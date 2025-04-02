use std::cmp::min;
use std::vec;

use crate::cache_client::{CacheClient, CacheResult};
use crate::ann_index::ANNIndex;
use crate::types::EmbeddingData;
use crate::utils::get_unix_seconds;

pub const EMBEDDING_DIMENSION: usize = 1536;

pub struct Cache<'a> {
    client: Box<dyn CacheClient>,
    ann_index: Box<dyn ANNIndex<'a>>,
    embedding_size: usize,
    index_initialized: bool,
    current_id: usize,
    cache_ttl: i64,
    redis_key: String,
}

impl <'a> Cache<'a> {
    pub fn new(client: Box<dyn CacheClient>, ann_index: Box<dyn ANNIndex<'a>>, redis_key: String, embedding_size: usize, cache_ttl: i64) -> Self {
        Self {
            client: client,
            ann_index: ann_index,
            embedding_size: embedding_size,
            index_initialized: false,
            current_id: 0,
            cache_ttl: cache_ttl,
            redis_key: redis_key,
        }
    }


    fn load_index(&mut self) -> CacheResult<()> {
        let data = self.get_all_embeddings()?;
        
        if data.len() == 0 {
            self.ann_index.init_index(1, EMBEDDING_DIMENSION)?;
            self.index_initialized = true;
            return Ok(());
        }
        
        if (data[0].embedding.len() == 0) {
            return Err("Something went wrong!".into());
        }
        
        self.ann_index.init_index(data.len(), EMBEDDING_DIMENSION)?;
        
        let max_val = data.iter().map(|d|(d.id)).max().unwrap();
        for d in data {
            self.ann_index.add_pt(d.embedding, d.id)?;
        }

        self.current_id = max_val + 1;
        self.index_initialized = true;

        Ok(())
    }


    pub fn store_embedding(&mut self, query: String, embedding: Vec<f32>, response: String) -> CacheResult<()>{
        if embedding.len() != self.embedding_size {
            return Err("embedding sizes mismatch!".into());
        }

        println!("Storing embedding");

        let new_embedding = embedding.clone();

        let id = self.current_id;

        let timestamp = get_unix_seconds();

        let data = EmbeddingData{id, query, embedding, response, timestamp };

        self.client.h_set(&self.redis_key, &id.to_string(), &serde_json::to_string(&data)?)?;
        
        if self.cache_ttl != 0 {
            self.client.expire(&self.redis_key,self.cache_ttl)?;
        }

        if (!self.index_initialized) {
            println!("Initializing index");
            self.ann_index.init_index(1000, EMBEDDING_DIMENSION)?;
            self.index_initialized = true;
        }

        let current_elements = self.ann_index.get_curr_ct()?;
        let max_elements = self.ann_index.get_max_elements()?;

        if (current_elements > max_elements) {
            println!("Resizing index");
            self.ann_index.resize(current_elements + 1000)?;
        }
        
        println!("Adding point to index");
        self.ann_index.add_pt(new_embedding, id)?;

        Ok(())
    }

    pub fn semantic_search(&mut self, embedding: &[f32], k: usize) -> CacheResult<Vec<EmbeddingData>> {
        if embedding.len() != self.embedding_size {
            return Err("embedding sizes mismatch!".into());
        }

        if !self.index_initialized {
            self.load_index()?;
        }

        let current_elements = self.ann_index.get_curr_ct()?;
        let adjusted_k = min(k, current_elements);

        if adjusted_k == 0 {
            return Ok(vec![]);
        }

        let result = self.ann_index.search_knn(embedding, adjusted_k)?;
        let ids: Vec<String> = result.iter().map(|n| n.0.to_string()).collect();        
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let data: Vec<Option<String>> = self.client.hm_get(&self.redis_key, &id_refs)?;

        let mut parsed_data = Vec::new();
        for maybe_json in data {
            if let Some(json_str) = maybe_json {
                let embedding_data: EmbeddingData = serde_json::from_str(&json_str)?;
                parsed_data.push(embedding_data);
            }
        }

        return Ok(parsed_data);
    }

    pub fn get_all_embeddings(&mut self) -> CacheResult<Vec<EmbeddingData>> {
        let data = self.client.h_get_all(&self.redis_key)?;

        let embeddings = data.into_iter()
            .map(|(_, v)| serde_json::from_str(&v))
            .filter_map(Result::ok)
            .collect();

        // Do not need to remap embeddings back into numbers?
        
        return Ok(embeddings);
    }

}

fn mapper((_, v): (String, String)) -> CacheResult<EmbeddingData> {
    let embedding_data: EmbeddingData = serde_json::from_str(&v)?;
    return Ok(embedding_data);
}
