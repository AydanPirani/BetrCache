use std::time::SystemTime;

use crate::{SIMILARITY_THRESHOLD, THRESHOLD};
use crate::api::{EmbeddingOptions, GPTOptions, get_embedding, get_gpt_response};
use crate::cache::Cache;
use crate::similarity::cosine_similarity;

pub fn get_unix_seconds() -> u64 {
    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(n) => n.as_secs(),
        Err(_) => panic!("SystemTime before UNIX EPOCH!"),
    }
}

pub async fn query<'a>(
    prompt: &str,
    gpt_options: &GPTOptions<'a>,
    embedding_options: &EmbeddingOptions<'a>,
    cache: &mut Cache<'a>,
) -> Result<String, Box<dyn std::error::Error>> {
    let query_embedding = get_embedding(prompt, &embedding_options).await?;

    let candidates = cache.semantic_search(&query_embedding, *THRESHOLD)?;

    if candidates.len() == 0 {
        println!("No candidates found! Querying API..");
        let response = get_gpt_response(prompt, &gpt_options).await?;
        cache.store_embedding(prompt.to_string(), query_embedding, response.clone())?;
        return Ok(response);
    }

    let mut best_candidate = 0;
    let mut best_similarity =
        cosine_similarity(&candidates[best_candidate].embedding, &query_embedding);

    for idx in 1..candidates.len() {
        let similarity = cosine_similarity(&candidates[idx].embedding, &query_embedding);

        if similarity > best_similarity {
            best_candidate = idx;
            best_similarity = similarity;
        }
    }

    if best_similarity > *SIMILARITY_THRESHOLD {
        println!("Cache hit! Returning cached response..");
        return Ok(candidates[best_candidate].response.clone());
    } else {
        println!("No good candidates found! Querying API..");
        let response = get_gpt_response(prompt, &gpt_options).await?;
        cache.store_embedding(prompt.to_string(), query_embedding, response.clone())?;
        return Ok(response);
    }
}
