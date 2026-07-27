from __future__ import annotations

import asyncio
import os
import time
import unittest

os.environ.setdefault("APP_SESSION_SECRET", "event-loop-test-session-secret-123456789")

from server.core.blocking import blocking_work_pool


class EventLoopIsolationTests(unittest.TestCase):
    def test_blocking_worker_keeps_heartbeat_running(self) -> None:
        async def scenario() -> int:
            heartbeat_count = 0

            async def heartbeat() -> None:
                nonlocal heartbeat_count
                for _ in range(12):
                    await asyncio.sleep(0.01)
                    heartbeat_count += 1

            await asyncio.gather(
                heartbeat(),
                blocking_work_pool.run("artifact", time.sleep, 0.1),
            )
            return heartbeat_count

        self.assertGreaterEqual(asyncio.run(scenario()), 8)


if __name__ == "__main__":
    unittest.main()
