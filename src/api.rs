use reqwest::Client;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use serde_json::Value;
use std::error::Error;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Provider {
    OpenAI,
    OpenRouter,
}

pub struct GPTOptions<'a> {
    pub model: &'a str,
    pub provider: Provider,
    pub api_key: &'a str,
    pub prefix: String,
}

// New options struct for embeddings.
pub struct EmbeddingOptions<'a> {
    pub model: &'a str,
    pub provider: Provider,
    pub api_key: &'a str,
}

#[derive(Serialize)]
struct OpenAIRequest<'a> {
    input: &'a str,
    model: &'a str,
}

#[derive(Deserialize)]
pub struct OpenAIResponse {
    pub data: Vec<OpenAIEmbeddingData>,
}

#[derive(Deserialize)]
pub struct OpenAIEmbeddingData {
    pub embedding: Vec<f64>,
}

pub async fn get_gpt_response<'a>(prompt: &str, options: &GPTOptions<'a>) -> Result<String, Box<dyn Error>> {
    let client = Client::new();

    // Choose the endpoint and API key based on the selected provider.
    let (url, api_key) = match options.provider {
        Provider::OpenAI => (
            "https://api.openai.com/v1/chat/completions",
            options.api_key
        ),
        Provider::OpenRouter => (
            "https://openrouter.ai/api/v1/chat/completions",
            options.api_key
        ),
    };

    let content = options.prefix.clone() + prompt;
    println!("Prompt: {}", content);

    let response = client.post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": options.model,
            "messages": [{ "role": "user", "content": content }]
        }))
        .send()
        .await?;

    let response_json: Value = response.json().await?;
    // println!("Full JSON Response: {:?}", response_json); // Debugging line

    let content = response_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();

    Ok(content)
}


// New function for obtaining embeddings.
pub async fn get_embedding<'a>(input: &str, options: &EmbeddingOptions<'a>) -> Result<Vec<f32>, Box<dyn Error>> {
    let client = Client::new();

    if Provider::OpenRouter == options.provider {
        return Err("OpenRouter embeddings not implemented yet".into());
    }

    let url="https://api.openai.com/v1/embeddings";

    let payload = json!({
        "input": input,
        "model": options.model,
    });

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", options.api_key))
        .json(&payload)
        .send()
        .await?;

    let response_json: Value = response.json().await?;
    // println!("Full JSON Response for embedding: {:?}", response_json); // Debugging line

    // Extract the embedding vector.
    let embedding = response_json["data"][0]["embedding"]
        .as_array()
        .ok_or("Invalid embedding response structure")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect::<Vec<f32>>();

    Ok(embedding)
}