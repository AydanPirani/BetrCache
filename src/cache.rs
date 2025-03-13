use crate::cache_client::CacheClient;
use crate::ann_index::{self, ANNIndex};

pub struct Cache<'a> {
    client: Box<dyn CacheClient>,
    ann_index: Box<dyn ANNIndex<'a>>,

}

impl <'a> Cache<'a> {
    pub fn new(&mut self, client: Box<dyn CacheClient>, ann_index: Box<dyn ANNIndex<'a>>) -> () {
        self.client = client;
        self.ann_index = ann_index;
        return;
    }
}