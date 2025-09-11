import random

from models.player import GameContext, Item, Player, PlayerSnapshot


class RandomPausePlayer(Player):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:  # noqa: F821
		super().__init__(snapshot, ctx)

	def propose_item(self, history: list[Item]) -> Item | None:
		if random.random() < 0.75:
			return random.choice(self.memory_bank)
		return None
