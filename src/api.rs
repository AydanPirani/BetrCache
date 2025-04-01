use reqwest::Client;
use serde_json::json;
use serde_json::Value;
use std::error::Error;

#[derive(Debug)]
pub enum Provider {
    OpenAI,
    OpenRouter,
}

pub struct GPTOptions {
    pub model: String,
    pub provider: Provider,
    pub openai_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
}

pub struct API;

impl API {
    pub async fn get_gpt_response(prompt: &str, options: &GPTOptions) -> Result<String, Box<dyn Error>> {
        let client = Client::new();

        // Choose the endpoint and API key based on the selected provider.
        let (url, api_key) = match options.provider {
            Provider::OpenAI => (
                "https://api.openai.com/v1/chat/completions",
                options.openai_api_key.as_ref().expect("Missing OPENAI_API_KEY")
            ),
            Provider::OpenRouter => (
                "https://openrouter.ai/api/v1/chat/completions",
                options.openrouter_api_key.as_ref().expect("Missing OPENROUTER_API_KEY")
            ),
        };

        let response = client.post(url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&json!({
                "model": options.model,
                "messages": [{ "role": "user", "content": prompt }]
            }))
            .send()
            .await?;

        let response_json: Value = response.json().await?;
        println!("Full JSON Response: {:?}", response_json); // Debugging line

        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        Ok(content)
    }
}
