mod cache;
mod cache_client;
mod ann_index;
mod types;
mod utils;
mod similarity;

use cache_client::CacheClient;
use cache_client::RedisClient;

fn main() {

    // Create a Redis client instance, handle error if occurs
    let mut client = RedisClient::new().expect("Failed to create Redis client");

    // Example usage
    client.h_set("user:1", "name", "Alice").expect("Failed to set value in Redis");
    let value = client.hm_get("user:1", &["name"]).expect("Failed to get value from Redis");

    println!("Retrieved value: {:?}", value);

}
