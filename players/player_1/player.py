from collections import Counter
from uuid import UUID

from models.player import GameContext, Item, Player, PlayerSnapshot


class Player1(Player):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

		self.subj_pref_ranking = {
			subject: snapshot.preferences.index(subject) for subject in snapshot.preferences
		}

		self.used_items: set[UUID] = set()

		# Adding dynamic playing style where we set the weights for coherence, importance and preference
		# based on the game context
		self.w_coh, self.w_imp, self.w_pref, self.w_nonmon = self._init_dynamic_weights(ctx, snapshot)

		# Print Player 1 ID and wait for input
		# print(f"Player 1 ID: {self.id}")
		# input("Press Enter to continue...")

	def propose_item(self, history: list[Item]) -> Item | None:
		# print('\nCurrent Memory Bank: ', self.memory_bank)
		# print('\nConversation History: ', history)

		# If history length is 0, return the first item from preferred sort ( Has the highest Importance)
		if len(history) == 0:
			memory_bank_imp = importance_sort(self.memory_bank)
			return memory_bank_imp[0] if memory_bank_imp else None

		# This Checks Repitition so we dont repeat any item that has already been said in the history, returns a filtered memory bank
		self._update_used_items(history)
		filtered_memory_bank = check_repetition(self.memory_bank, self.used_items)
		# print('\nCurrent Memory Bank: ', len(self.memory_bank))
		# print('\nFiltered Memory Bank: ', len(filtered_memory_bank))

		# Return None if there are no valid items to propose
		if len(filtered_memory_bank) == 0:
			memory_bank_imp = importance_sort(self.memory_bank)
			return memory_bank_imp[0] if memory_bank_imp else None

		coherence_scores = {
			item.id: coherence_check(item, history) for item in filtered_memory_bank
		}
		importance_scores = {item.id: item.importance for item in filtered_memory_bank}
		preference_scores = {
			item.id: score_item_preference(item.subjects, self.subj_pref_ranking)
			for item in filtered_memory_bank
		}
		nonmonotonousness_scores = {
			item.id: score_nonmonotonousness(item, history) for item in filtered_memory_bank
		}

		item = choose_item(
			filtered_memory_bank,
			coherence_scores,
			importance_scores,
			preference_scores,
			nonmonotonousness_scores,
			weights=(self.w_coh, self.w_imp, self.w_pref, self.w_nonmon),
		)

		if item:
			return item
		else:
			return None

	def _update_used_items(self, history: list[Item]) -> None:
		# Update the used_items set with items from history
		self.used_items.update(item.id for item in history)

	def _init_dynamic_weights(
		self, ctx: GameContext, snapshot: PlayerSnapshot
	) -> tuple[float, float, float]:
		P = ctx.number_of_players
		L = ctx.conversation_length
		S = len(snapshot.preferences)
		B = len(snapshot.memory_bank)

		# Base Weights
		w_coh, w_imp, w_pref, w_nonmon = 0.4, 0.3, 0.2, 0.1

		# Length of Conversation
		if L <= 12:
			# short: focus importance
			w_coh, w_imp, w_pref, w_nonmon = 0.3, 0.45, 0.2, 0.05
		elif L >= 31:
			# long: focus coherence even more strongly
			w_coh, w_imp, w_pref, w_nonmon = 0.5, 0.2, 0.15, 0.15

		# Player Size
		if P <= 3:
			# small: More control, nudge coherence
			w_coh += 0.05
			w_imp -= 0.05
		elif P >= 6:
			# large: Less control, bank importance more heavily and cut preference
			w_coh -= 0.1
			w_imp += 0.1
			w_pref = max(w_pref - 0.05, 0.1)

		# Subject length
		if S <= 6:
			# More Overlap: coherence is easier so focus importance
			w_imp += 0.05
			w_coh -= 0.05
		elif S >= 12:
			# Less Overlap: harder to hit coherence, but valuable when possible
			w_coh += 0.10
			w_imp -= 0.05
			w_pref = max(w_pref - 0.05, 0.10)

		# Inventory Length
		if B <= 8:
			# conservative, focus coherence
			w_coh += 0.05
			w_imp -= 0.05
		elif B >= 16:
			# Less Conservative, focus importance
			w_imp += 0.05
			w_coh -= 0.05

		# clamp to [0,1] and softly renormalize to keep sum≈1
		w_coh = max(0.0, min(1.0, w_coh))
		w_imp = max(0.0, min(1.0, w_imp))
		w_pref = max(0.0, min(1.0, w_pref))
		w_nonmon = max(0.0, min(1.0, w_nonmon))

		total = w_coh + w_imp + w_pref + w_nonmon
		if total > 0:
			w_coh, w_imp, w_pref, w_nonmon = (w_coh / total, w_imp / total, w_pref / total, w_nonmon / total)

		# Cap preference weight depending on conversation length
		if L <= 12 and w_pref > 0.18:
			w_pref = 0.18
		elif L >= 31 and w_pref > 0.15:
			w_pref = 0.15

		# Renormalize after capping preference
		total = w_coh + w_imp + w_pref + w_nonmon
		if total > 0:
			w_coh, w_imp, w_pref, w_nonmon = (w_coh / total, w_imp / total, w_pref / total, w_nonmon / total)

		return (w_coh, w_imp, w_pref, w_nonmon)


# Helper Functions #


def check_repetition(memory_bank: list[Item], used_items: set[UUID]) -> list[Item]:
	# Filter out items with IDs already in the used_items set
	return [item for item in memory_bank if item.id not in used_items]


def coherence_check(current_item: Item, history: list[Item]) -> float:
	# Check the last 3 items in history (or fewer if history is shorter)
	recent_history = []
	start_idx = max(0, len(history) - 3)

	for i in range(len(history) - 1, start_idx - 1, -1):
		item = history[i]
		if item is None:
			break
		recent_history.append(item)

	# Count occurrences of each subject in the recent history
	subject_count = Counter()
	for item in recent_history:
		subject_count.update(item.subjects)

	# See if all subjects in the current item are appear once or twice in the history
	subjects = current_item.subjects
	counts = [subject_count.get(s, 0) for s in subjects]

	if any(c == 0 for c in counts):
		return 0.0
	
	if all(c >= 2 for c in counts):
		return 1.0 # awarding full point for 2 mentions
	
	if all(c >= 1 for c in counts):
		return 0.5
	
	return 0.0

	# Debugging prints
	# print("\nCurrent Item Subjects:", current_item.subjects)
	# print("History Length:", len(history))
	# print("Recent History:", [item.subjects for item in recent_history])
	# print("Subject Count:", subject_count)
	# print("Coherence Score Before Normalization:", coherence_score)
	# print("Coherence Score After Normalization:", coherence_score / len(current_item.subjects) if current_item.subjects else 0.0)
	# print("Number of Subjects in Current Item:", len(current_item.subjects))


def score_nonmonotonousness(current_item: Item, history: list[Item]) -> float:
	recent_history = history[-3:]
	penalty = 0

	for subj in current_item.subjects:
		if all(any(prev_subj == subj for prev_subj in prev_item.subjects) for prev_item in recent_history):
			penalty -= 1

	if current_item in history:
		penalty -= 1

	max_penalty = len(current_item.subjects) + 1 if current_item.subjects else 1

	score = 1.0 - (penalty / max_penalty)
	return max(0.0, score)


def coherence_sort(memory_bank: list[Item], history: list[Item]) -> list[Item]:
	# Sort the memory bank based on coherence scores in descending order
	# use a lambda on each item to check coherence score
	sorted_memory = sorted(
		memory_bank, key=lambda item: coherence_check(item, history), reverse=True
	)
	return sorted_memory


def importance_sort(memory_bank: list[Item]) -> list[Item]:
	# Sort the memory bank based on the importance attribute in descending order
	return sorted(memory_bank, key=lambda item: item.importance, reverse=True)


def score_item_preference(subjects, subj_pref_ranking):
	if not subjects:
		return 0.0

	S_length = len(subj_pref_ranking)
	bonuses = [
		1 - subj_pref_ranking.get(subject, S_length) / S_length for subject in subjects
	]  # bonus is already a preference score of sorts
	return sum(bonuses) / len(bonuses)


def calculate_weighted_score(
	item_id, coherence_scores, importance_scores, preference_scores, nonmonotonousness_scores, weights
):
	w1, w2, w3, w4 = weights
	coherence = coherence_scores.get(item_id, 0.0)
	importance = importance_scores.get(item_id, 0.0)
	preference = preference_scores.get(item_id, 0.0)
	nonmonotonousness = nonmonotonousness_scores.get(item_id, 0.0)

	return w1 * coherence + w2 * importance + w3 * preference + w4 * nonmonotonousness


def choose_item(
	memory_bank: list[Item],
	coherence_scores: dict[UUID, float],
	importance_scores: dict[UUID, float],
	preference_scores: dict[UUID, float],
	nonmonotonousness_scores: dict[UUID, float],
	weights: tuple[float, float, float, float],
):
	weighted_item_scores = {
		item: calculate_weighted_score(
			item.id, coherence_scores, importance_scores, preference_scores, nonmonotonousness_scores, weights
		)
		for item in memory_bank
	}

	sorted_items = sorted(weighted_item_scores.items(), key=lambda item: item[1], reverse=True)
	return sorted_items[0][0] if sorted_items else None

	# Takes in the total memory bank and scores each item based on whatever weighting system we have
	# Actually should make this a function in the class so it can have access to the contributed items/memory bank
	# Should automatically score things that were already in the contributed items a 0

	# As its scored, add it to a set thats sorted by the score. Return Set
