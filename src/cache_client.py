from abc import ABC, abstractmethod
import redis


class CacheClient(ABC):
    @abstractmethod
    def h_set(self, key: str, field: str, value: str) -> None:
        ...

    @abstractmethod
    def hm_get(self, key: str, fields: list[str]) -> list[str | None]:
        ...

    @abstractmethod
    def h_get_all(self, key: str) -> dict[str, str]:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def expire(self, key: str, seconds: int) -> None:
        ...


class RedisClient(CacheClient):
    def __init__(self, url: str):
        self.client = redis.Redis.from_url(url)

    def h_set(self, key: str, field: str, value: str) -> None:
        self.client.hset(key, field, value)

    def hm_get(self, key: str, fields: list[str]) -> list[str | None]:
        return self.client.hmget(key, fields)

    def h_get_all(self, key: str) -> dict[str, str]:
        return self.client.hgetall(key)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def expire(self, key: str, seconds: int) -> None:
        self.client.expire(key, seconds)
