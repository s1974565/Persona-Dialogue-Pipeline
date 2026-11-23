import asyncio
import collections


class PromptTracker:
    """
    Tracks the call count for recent prompts using an LRU (Least Recently Used) strategy.
    """

    def __init__(self, max_size=1000):
        self.prompt_counts = collections.OrderedDict()

        self.max_size = max_size
        self.lock = asyncio.Lock()

    async def get_and_increment_count(self, prompt: str):
        """
        Increments the count for a given prompt and returns the old count.
        If the cache is full, the least recently used prompt is removed.
        """
        async with self.lock:
            count = self.prompt_counts.get(prompt, 0)
            self.prompt_counts[prompt] = count + 1

            # Mark this prompt as the most recently used by moving it to the end.
            self.prompt_counts.move_to_end(prompt)
            # If the cache has exceeded its max size, remove the oldest item (the first entry).
            if len(self.prompt_counts) > self.max_size:
                self.prompt_counts.popitem(last=False)

            return count