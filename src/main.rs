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
use api::{GPTOptions, Provider, EmbeddingOptions};
use tokio;

use std::env;
use std::io;
use std::io::Write;
use dotenv::dotenv;
use ann_index::HnswAnnIndex;
use lazy_static::lazy_static;

use crate::utils::query;

lazy_static! {
    static ref REDIS_URL: String = env::var("REDIS_URL").expect("REDIS_URL must be set");
    static ref LLM_MODEL: String = env::var("LLM_MODEL").expect("LLM_MODEL must be set");
    static ref EMBEDDINGS_MODEL: String = env::var("EMBEDDINGS_MODEL").expect("EMBEDDINGS_MODEL must be set");
    static ref OPENAI_API_KEY: String = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    static ref OPENROUTER_API_KEY: String = env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");
    
    static ref EMBEDDINGS_PROVIDER: Provider = Provider::OpenAI;
    static ref LLM_PROVIDER: Provider = Provider::OpenRouter;

    static ref THRESHOLD: usize = 5;
    static ref SIMILARITY_THRESHOLD: f32 = 0.8;
    static ref EMBEDDING_DIMENSION: usize = 1536;
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // let redis_url = env::var("REDIS_URL")?.clone();
    // let llm_model = env::var("LLM_MODEL")?.clone();
    // let embeddings_model = env::var("EMBEDDINGS_MODEL")?.clone();
    
    // let openai_api_key = env::var("OPENAI_API_KEY")?.clone();
    // let openrouter_api_key = env::var("OPENROUTER_API_KEY")?.clone();

    // let embeddings_provider = Provider::OpenAI;

    let llm_key: &str = match *LLM_PROVIDER {
        Provider::OpenAI => &OPENAI_API_KEY,
        Provider::OpenRouter => &OPENROUTER_API_KEY,
    };

    let embeddings_key: &str = match *EMBEDDINGS_PROVIDER {
        Provider::OpenAI => &OPENAI_API_KEY,
        Provider::OpenRouter => &OPENROUTER_API_KEY,
    };

    let gpt_options = GPTOptions {
        model: &LLM_MODEL,
        provider: *LLM_PROVIDER,
        api_key: llm_key,
        prefix: "You are a search assistant. Give me a response in 5 sentences.".to_string(),
    };

    let embedding_options = EmbeddingOptions {
        model: &EMBEDDINGS_MODEL,
        provider: *EMBEDDINGS_PROVIDER,
        api_key: embeddings_key,
    };

    let index = HnswAnnIndex::new(1000, *EMBEDDING_DIMENSION);
    let mut client = RedisClient::new(&REDIS_URL)?;
    client.delete("embeddings")?;
    
    let mut cache: Cache<'_> = Cache::new(Box::new(client), Box::new(index), "embeddings".to_string(), *EMBEDDING_DIMENSION, 0);
    
    loop {
        print!("> ");
        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                // Trim the input to remove any trailing newline characters
                let trimmed = input.trim();
                println!("You entered: {}", trimmed);
                let res = query(trimmed, &gpt_options, &embedding_options, &mut cache).await;
                match res {
                    Ok(response) => println!("Response: {}", response),
                    Err(err) => eprintln!("Error: {}", err),
                }
            }
            Err(e) => {
                println!("Error reading input: {}", e);
                break;
            }
        }
    }
    
    Ok(())
}