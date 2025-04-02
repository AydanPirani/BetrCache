mod ann_index;
mod api;
mod cache_client;
mod cache;
mod embeddings;
mod index;
mod similarity;
mod types;
mod utils;

use cache::Cache;
use cache_client::CacheClient;
use cache_client::RedisClient;
use api::{GPTOptions, Provider, EmbeddingOptions, get_embedding};
use tokio;

use std::env;
use dotenv::dotenv;
use ann_index::HnswAnnIndex;
use cache::DIMENSION;



#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let redis_url = env::var("REDIS_URL")?;
    let llm_model = env::var("LLM_MODEL")?;
    let embeddings_model = env::var("EMBEDDINGS_MODEL")?;
    
    let openai_api_key = env::var("OPENAI_API_KEY")?;
    let openrouter_api_key = env::var("OPENROUTER_API_KEY")?;

    let llm_provider = Provider::OpenRouter;
    let embeddings_provider = Provider::OpenAI;

    let llm_key = match llm_provider {
        Provider::OpenAI => openai_api_key.as_str(),
        Provider::OpenRouter => openrouter_api_key.as_str(),
    };

    let embeddings_key = match embeddings_provider {
        Provider::OpenAI => openai_api_key.as_str(),
        Provider::OpenRouter => openrouter_api_key.as_str(),
    };

    let gpt_options = GPTOptions {
        model: llm_model.as_str(),
        provider: llm_provider,
        api_key: llm_key,
        prefix: "You are a search assistant. Give me a response in 5 sentences.".to_string(),
    };

    let embedding_options = EmbeddingOptions {
        model: embeddings_model.as_str(),
        provider: embeddings_provider,
        api_key: embeddings_key,
    };

    let index = HnswAnnIndex::new(1000, DIMENSION);
    let client = RedisClient::new(redis_url)?;
    
    let cache: Cache<'_> = Cache::new(Box::new(client), Box::new(index), "embeddings".to_string(), DIMENSION, 10);
    
    let prompt = "What is Chicago known for?";
    let embeddings = get_embedding(prompt, &embedding_options).await?;
    println!("Embeddings: {:?}", embeddings);

    // cache.store_embedding(prompt.to_string(), embeddings, "response".to_string())?;
    
    // let res = API::get_gpt_response(prompt, &options).await;
    // match res {
    //     Ok(response) => println!("GPT Response: {}", response),
    //     Err(err) => eprintln!("Error: {}", err),
    // };
    


    // Example usage
    // client
    //     .h_set("user:1", "name", "Alice")
    //     .expect("Failed to set value in Redis");

    // let value = client
    //     .hm_get("user:1", &["name"])
    //     .expect("Failed to get value from Redis");
    // println!("Retrieved value: {:?}", value);
    
    Ok(())
}