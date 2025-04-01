mod ann_index;
mod api;
mod cache_client;
mod cache;
mod index;
mod similarity;
mod types;
mod utils;

use cache_client::CacheClient;
use cache_client::RedisClient;
use api::{API, GPTOptions, Provider};
use tokio;

use std::env;
use dotenv::dotenv;



#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let url = env::var("REDIS_URL")?;
    let model = env::var("MODEL")?;
    
    let openai_api_key = env::var("OPENAI_API_KEY").ok();
    let openrouter_api_key = env::var("OPENROUTER_API_KEY").ok();

    let provider = Provider::OpenRouter;

    let options = GPTOptions {
        model,
        provider,
        openai_api_key,
        openrouter_api_key,
        prefix: "You are a search assistant. Give me a response in 5 sentences.".to_string(),
    };

    let prompt = "What is Chicago known for?";
 
    let res = API::get_gpt_response(prompt, &options).await;
    match res {
        Ok(response) => println!("GPT Response: {}", response),
        Err(err) => eprintln!("Error: {}", err),
    };

    // Create a Redis client instance, handle error if occurs
    let mut client = RedisClient::new(url).expect("Failed to create Redis client");

    // Example usage
    client
        .h_set("user:1", "name", "Alice")
        .expect("Failed to set value in Redis");

    let value = client
        .hm_get("user:1", &["name"])
        .expect("Failed to get value from Redis");

    println!("Retrieved value: {:?}", value);
    
    Ok(())
}