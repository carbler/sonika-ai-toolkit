"""RedisTool — basic Redis key-value operations."""

import json
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

_VALID_ACTIONS = {"get", "set", "delete", "exists", "expire", "keys"}


class _RedisInput(BaseModel):
    action: str = Field(
        description=(
            "Operation to perform: 'get', 'set', 'delete', 'exists', 'expire', 'keys'. "
            "'keys' returns all keys matching a pattern (use '*' for all)."
        )
    )
    key: str = Field(description="Redis key (or glob pattern for 'keys' action).")
    value: str = Field(default="", description="Value to store (only for 'set').")
    ttl: int = Field(default=0, description="TTL in seconds for 'set' (0 = no expiry) or 'expire'.")
    host: str = Field(default="localhost", description="Redis host. Default 'localhost'.")
    port: int = Field(default=6379, description="Redis port. Default 6379.")
    password: str = Field(default="", description="Redis password (optional).")
    db: int = Field(default=0, description="Redis database number. Default 0.")


class RedisTool(BaseTool):
    name: str = "redis_operation"
    description: str = (
        "Perform Redis operations: get, set, delete, exists, expire, keys. "
        "Useful for caching, session storage, and pub-sub patterns."
    )
    args_schema: Type[BaseModel] = _RedisInput
    risk_hint: int = 1

    def _run(
        self,
        action: str,
        key: str,
        value: str = "",
        ttl: int = 0,
        host: str = "localhost",
        port: int = 6379,
        password: str = "",
        db: int = 0,
    ) -> str:
        try:
            import redis
        except ImportError:
            return "Error: redis not installed. Run: pip install redis"

        action = action.lower()
        if action not in _VALID_ACTIONS:
            return f"Error: unknown action '{action}'. Valid actions: {', '.join(sorted(_VALID_ACTIONS))}"

        try:
            client = redis.Redis(
                host=host,
                port=port,
                password=password or None,
                db=db,
                decode_responses=True,
            )
            if action == "get":
                result = client.get(key)
                return result if result is not None else "(nil)"
            elif action == "set":
                if ttl > 0:
                    client.setex(key, ttl, value)
                else:
                    client.set(key, value)
                return "OK"
            elif action == "delete":
                deleted = client.delete(key)
                return f"Deleted {deleted} key(s)"
            elif action == "exists":
                return str(bool(client.exists(key)))
            elif action == "expire":
                if ttl <= 0:
                    return "Error: 'expire' requires ttl > 0"
                client.expire(key, ttl)
                return "OK"
            elif action == "keys":
                keys = client.keys(key)
                return json.dumps(keys, ensure_ascii=False)
        except Exception as e:
            return f"Error: {e}"
