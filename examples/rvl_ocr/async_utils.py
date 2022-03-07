import asyncio
import logging
import time
from asyncio import Semaphore

logger = logging.getLogger(__name__)


class SemaphoreWithDelay(Semaphore):
    def __init__(self, value=1, delay=3, *, loop=None):
        super().__init__(value=value, loop=loop)
        self.delay = delay
        self.last_aquired = time.time()
        self.time_check_semaphore = Semaphore(1)

    def time_to_wait(self):
        return max(self.last_aquired + self.delay - time.time(), 0)

    async def wait_since_last(self):
        async with self.time_check_semaphore:
            to_wait = self.time_to_wait()
            logging.debug(f"Awaiting {to_wait} before allowing semaphore entry")
            await asyncio.sleep(to_wait)
            self.last_aquired = time.time()

    async def acquire(self):
        res = await super().acquire()
        await self.wait_since_last()
        return res
