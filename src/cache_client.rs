use redis::{Commands, Connection};
use std::env;

pub type CacheResult<T> = Result<T, Box<dyn std::error::Error>>;

pub trait CacheClient {
    fn create(&mut self) -> CacheResult<()>;
    fn connect(&mut self) -> CacheResult<()>;
    fn h_set(&mut self, key: &str, field: &str, value: &str) -> CacheResult<()>;
    fn hm_get(&mut self, key: &str, fields: &[&str] ) -> CacheResult<Vec<Option<String>>>;
    fn h_get_all(&mut self, key: &str) -> CacheResult<std::collections::HashMap<String, String>>;
    fn delete(&mut self, key: &str) -> CacheResult<()>;
    fn expire(&mut self, key: &str, seconds: i64) -> CacheResult<()>;
}

pub struct RedisClient {
    url: String,
    con: Connection,
}

impl CacheClient for RedisClient {
    fn create(&mut self) -> CacheResult<()> {
        self.url = env::var("REDIS_URL")?;
        return Ok(());
    }

    fn connect(&mut self) -> CacheResult<()> {
        let client = redis::Client::open(self.url.clone())?;
        self.con = client.get_connection()?;
        return Ok(());
    }

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