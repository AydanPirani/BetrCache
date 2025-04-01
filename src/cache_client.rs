use redis::{Commands, Connection};
use std::env;
use dotenv::dotenv;

pub type CacheResult<T> = Result<T, Box<dyn std::error::Error>>;

pub trait CacheClient {
    fn h_set(&mut self, key: &str, field: &str, value: &str) -> CacheResult<()>;
    fn hm_get(&mut self, key: &str, fields: &[&str] ) -> CacheResult<Vec<Option<String>>>;
    fn h_get_all(&mut self, key: &str) -> CacheResult<std::collections::HashMap<String, String>>;
    fn delete(&mut self, key: &str) -> CacheResult<()>;
    fn expire(&mut self, key: &str, seconds: i64) -> CacheResult<()>;
}

pub struct RedisClient {
    con: Connection,
}

impl RedisClient {
    /// Constructor for RedisClient
    pub fn new() -> CacheResult<Self> {
        println!("Creating Redis Client");

        // Load environment variables from the .env file
        dotenv().ok();

        let url = env::var("REDIS_URL")?; // Get Redis URL from env
        println!("Connecting to Redis at: {}", url);
        let client = redis::Client::open(url.clone())?;
        let con = client.get_connection()?; // Establish connection

        Ok(Self { con }) // Return an initialized RedisClient
    }
}

impl CacheClient for RedisClient {
    fn h_set(&mut self, key: &str, field: &str, value: &str) -> CacheResult<()> {
        // Throw away result, just ensure that it does not fail
        let _: () = self.con.hset(key, field, value)?;
        return Ok(());
    }

    fn hm_get(&mut self, key: &str, fields: &[&str] ) -> CacheResult<Vec<Option<String>>> {
        let res = self.con.hget(key, fields)?;
        return Ok(res);
    }

    fn h_get_all(&mut self, key: &str) -> CacheResult<std::collections::HashMap<String, String>> {
        let res = self.con.hgetall(key)?; 
        return Ok(res);
    }

    fn delete(&mut self, key: &str) -> CacheResult<()> {
        let _: () = self.con.del(key)?;
        return Ok(());
    }

    fn expire(&mut self, key: &str, seconds: i64) -> CacheResult<()> {
        let _: () = self.con.expire(key, seconds)?;
        return Ok(());
    }
}