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
		self.w_coh, self.w_imp, self.w_pref = self._init_dynamic_weights(ctx, snapshot)

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

		item = choose_item(
			filtered_memory_bank,
			coherence_scores,
			importance_scores,
			preference_scores,
			weights=(self.w_coh, self.w_imp, self.w_pref),
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
		w_coh, w_imp, w_pref = 0.45, 0.35, 0.20

		# Length of Conversation
		if L <= 12:
			# short: focus importance
			w_coh, w_imp, w_pref = 0.35, 0.50, 0.15
		elif L >= 31:
			# long: focus coherence even more strongly
			w_coh, w_imp, w_pref = 0.65, 0.20, 0.15

		# Player Size
		if P <= 3:
			# small: More control, nudge coherence
			w_coh += 0.05
			w_imp -= 0.05
		elif P >= 6:
			# large: Less control, bank importance more heavily and cut preference
			w_coh -= 0.10
			w_imp += 0.10
			w_pref = max(w_pref - 0.05, 0.10)

		# Subject length
		if S <= 6:
			# More Overlap: coherence is easier so focus importance
			w_imp += 0.05
			w_coh -= 0.05
		elif S >= 12:
			# Less Overlap: harder to hit coherence, but valuable when possible
			w_coh += 0.10
			w_imp -= 0.05
			w_pref -= 0.05

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

		total = w_coh + w_imp + w_pref
		if total > 0:
			w_coh, w_imp, w_pref = (w_coh / total, w_imp / total, w_pref / total)

		# Cap preference weight depending on conversation length
		if L <= 12 and w_pref > 0.18:
			w_pref = 0.18
		elif L >= 31 and w_pref > 0.15:
			w_pref = 0.15

		# Renormalize after capping preference
		total = w_coh + w_imp + w_pref
		if total > 0:
			w_coh, w_imp, w_pref = (w_coh / total, w_imp / total, w_pref / total)

		return (w_coh, w_imp, w_pref)


# Helper Functions #


def check_repetition(memory_bank: list[Item], used_items: set[UUID]) -> list[Item]:
	# Filter out items with IDs already in the used_items set
	return [item for item in memory_bank if item.id not in used_items]


def coherence_check(current_item: Item, history: list[Item]) -> float:
	# Check the last 3 items in history (or fewer if history is shorter)
	recent_history = history[-3:]
	coherence_score = 0

	# Count occurrences of each subject in the recent history
	subject_count = {}
	for item in recent_history:
		for subject in item.subjects:
			subject_count[subject] = subject_count.get(subject, 0) + 1

	has_missing = False
	all_twice = True

	# See if all subjects in the current item are appear once or twice in the history
	for subject in current_item.subjects:
		count = subject_count.get(subject, 0)

		if count != 2:
			if count == 0:
				has_missing = True
			else:
				all_twice = False
			break

		# if subject_count.get(subject, 0) in [1, 2]:
		# 	coherence_score += 1

	if has_missing:
		coherence_score -= 1  # penalize if subject is missing from prior context. can refine later
	elif all_twice:
		coherence_score += 1  # reward if all subjects are mentioned exactly twice in prior context
	else:
		coherence_score = 0

	# Debugging prints
	# print("\nCurrent Item Subjects:", current_item.subjects)
	# print("History Length:", len(history))
	# print("Recent History:", [item.subjects for item in recent_history])
	# print("Subject Count:", subject_count)
	# print("Coherence Score Before Normalization:", coherence_score)
	# print("Coherence Score After Normalization:", coherence_score / len(current_item.subjects) if current_item.subjects else 0.0)
	# print("Number of Subjects in Current Item:", len(current_item.subjects))

	return (coherence_score + 1) / 2


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
	item_id, coherence_scores, importance_scores, preference_scores, weights
):
	w1, w2, w3 = weights
	coherence = coherence_scores.get(item_id, 0.0)
	importance = importance_scores.get(item_id, 0.0)
	preference = preference_scores.get(item_id, 0.0)

	return w1 * coherence + w2 * importance + w3 * preference


def choose_item(
	memory_bank: list[Item],
	coherence_scores: dict[UUID, float],
	importance_scores: dict[UUID, float],
	preference_scores: dict[UUID, float],
	weights: tuple[float, float, float],
):
	weighted_item_scores = {
		item: calculate_weighted_score(
			item.id, coherence_scores, importance_scores, preference_scores, weights
		)
		for item in memory_bank
	}

	sorted_items = sorted(weighted_item_scores.items(), key=lambda item: item[1], reverse=True)
	return sorted_items[0][0] if sorted_items else None

	# Takes in the total memory bank and scores each item based on whatever weighting system we have
	# Actually should make this a function in the class so it can have access to the contributed items/memory bank
	# Should automatically score things that were already in the contributed items a 0

	# As its scored, add it to a set thats sorted by the score. Return Set
