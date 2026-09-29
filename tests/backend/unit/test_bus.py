from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from traffic_intel.core.schemas import BusMessage
from traffic_intel.pipeline.bus import InMemoryEventBus, Topics


def _msg(topic: str, i: int) -> BusMessage:
    return BusMessage(topic=topic, timestamp=datetime.now(UTC), source="test", payload={"i": i})


async def test_fanout_and_topic_filter() -> None:
    bus = InMemoryEventBus()
    received: list[int] = []

    async def consume() -> None:
        async for m in bus.subscribe(Topics.RISK):
            received.append(m.payload["i"])
            if len(received) == 2:
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await bus.publish(_msg(Topics.RISK, 1))
    await bus.publish(_msg(Topics.INCIDENT, 99))
    await bus.publish(_msg(Topics.RISK, 2))
    await asyncio.wait_for(task, 1)
    assert received == [1, 2]
