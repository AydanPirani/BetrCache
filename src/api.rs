use reqwest::Client;
use serde_json::json;
use serde_json::Value;
use std::error::Error;
use std::env;
use dotenvy::dotenv;

pub struct API;

impl API {
    pub async fn get_gpt_response(prompt: &str, options: &GPTOptions) -> Result<String, Box<dyn Error>> {
        let client = Client::new();

        let response = client.post("https://api.openai.com/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", options.openai_api_key))
            .json(&json!({
                "model": options.model.as_deref().unwrap_or("gpt-4o-mini-2024-07-18"),
                "messages": [{ "role": "user", "content": prompt }]
            }))
            .send()
            .await?;

        let response_json: serde_json::Value = response.json().await?;
        println!("Full JSON Response: {:?}", response_json); // Debugging line
        
        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        Ok(content)
    }
}

pub struct GPTOptions {
    pub model: Option<String>,
    pub openai_api_key: String,
}

#[tokio::main]
async fn main() {
    dotenv().ok();

    let api_key = env::var("OPENAI_API_KEY").expect("Missing OPENAI_API_KEY in .env file");

    let options = GPTOptions {
        model: None,  // Default model will be used
        openai_api_key: api_key,
    };

    match API::get_gpt_response("Hello, GPT!", &options).await {
        Ok(response) => println!("GPT Response: {}", response),
        Err(err) => eprintln!("Error: {}", err),
    }
}
