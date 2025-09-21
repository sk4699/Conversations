from collections import Counter
from uuid import UUID

from models.player import GameContext, Item, Player, PlayerSnapshot
import os
from players.player_1.weight_policy import compute_initial_weights

class Player1(Player):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

		self.subj_pref_ranking = {
			subject: snapshot.preferences.index(subject) for subject in snapshot.preferences
		}

		self.used_items: set[UUID] = set()

		# Adding dynamic playing style where we set the weights for coherence, importance and preference
		# based on the game context
		# inside Player1.__init__
		(self.w_coh, self.w_imp, self.w_pref, self.w_nonmon, self.w_fresh, self.weighted, self.raw) = compute_initial_weights(
			ctx,
			snapshot,
			oracle_path=os.getenv("WEIGHTS_ORACLE_PATH", "players/player_1/data/weights_oracle_index.json"),
			alpha=0.7,
			nn_k=3,
		)

		print(f"Initial Weights: Coherence={self.w_coh:.3f}, Importance={self.w_imp:.3f}, Preference={self.w_pref:.3f}, Nonmonotonousness={self.w_nonmon:.3f}, Freshness={self.w_fresh:.3f}")
		# print(f"SUM: {sum((0.324405, 0.31929, 0.154073, 0.13617, 0.066061))}")
		self.ctx = ctx
		# Print Player 1 ID and wait for input
		# print(f"Player 1 ID: {self.id}")
		# input("Press Enter to continue...")

	def propose_item(self, history: list[Item]) -> Item | None:
		# print('\nCurrent Memory Bank: ', self.memory_bank)
		# print('\nConversation History: ', history)

		# If history length is 0, return the first item from preferred sort ( Has the highest Importance)
		# if len(history) == 0:
		# 	memory_bank_imp = importance_sort(self.memory_bank)
		# 	return memory_bank_imp[0] if memory_bank_imp else None

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
		importance_scores = {item.id: (item.importance, item.importance) for item in filtered_memory_bank}
		preference_scores = {
			item.id: score_item_preference(item.subjects, self.subj_pref_ranking)
			for item in filtered_memory_bank
		}
		nonmonotonousness_scores = {
			item.id: score_nonmonotonousness(item, history) for item in filtered_memory_bank
		}
		freshness_scores = {
			item.id: score_freshness(item, history) for item in filtered_memory_bank
		}

		score_sources = {"coherence": coherence_scores, "importance": importance_scores, "preference": preference_scores, "nonmonotonousness": nonmonotonousness_scores, "freshness": freshness_scores}

		# Checking for if it is a pause turn for the weighting system
		if len(history) !=0 and history[-1] is None:  # Last move was a pause
			# After a pause, we freshness to be weighted higher to take advantage of the opportunity
			self.w_coh, self.w_imp, self.w_pref, self.w_nonmon, self.w_fresh = (
				0.0,
				0.1,
				0.1,
				0.0,
				0.8,
			)

		best_item, best_now, weighted_scores = choose_item(
			self.weighted,
			self.raw,
			filtered_memory_bank,
			score_sources,
			weights=(self.w_coh, self.w_imp, self.w_pref, self.w_nonmon, self.w_fresh),
		)

		if best_item is None:
			return None

		# print("\nWeighted Item Scores:", max(weighted_scores.values(), default=None))

		# Decide to pause or speak
		if should_pause(
			history,
			best_item
		):
			# print('Decided to Pause')
			return None  # pause

		return best_item

	def _update_used_items(self, history: list[Item]) -> None:
		# Update the used_items set with items from history
		# if the item is None, it should not be added to the used_items set

		self.used_items.update(item.id for item in history if item is not None)


# Helper Functions #

def recent_subject_stats(history: list[Item], window: int = 6):
	# Look back `window` turns (skipping None), return:
	# - subj_counts: Counter of subjects in the window
	# - top_freq:    max frequency of any subject (0 if none)
	# - unique:      number of unique subjects
	# - seen_recent: set of subjects observed
	recent = [it for it in history[-window:] if it is not None]
	subjects = [s for it in recent for s in it.subjects]
	from collections import Counter
	subj_counts = Counter(subjects)
	top_freq = max(subj_counts.values()) if subj_counts else 0
	unique = len(subj_counts)
	return subj_counts, top_freq, unique, set(subjects)


def inventory_subjects(items: list[Item]) -> set[str]:
    #All subjects still available to play from the filtered memory bank.
	return {s for it in items for s in it.subjects}


def check_repetition(memory_bank: list[Item], used_items: set[UUID]) -> list[Item]:
	# Filter out items with IDs already in the used_items set
	return [item for item in memory_bank if item.id not in used_items]


def coherence_check(current_item: Item, history: list[Item]) -> float:
	# Check the last 3 items in history (or fewer if history is shorter)
	if current_item is None:
		raw_score = 0.0
		scaled_score = 0.0
		return raw_score, scaled_score

	recent_history = []
	start_idx = max(0, len(history) - 3)

	for i in range(len(history) - 1, start_idx - 1, -1):
		item = history[i]
		if item is None:
			break
		recent_history.append(item)

	# Count occurrences of each subject in the recent history
	subject_count = Counter()
	for item in recent_history:  # won't be None
		subject_count.update(item.subjects)

	# See if all subjects in the current item are appear once or twice in the history
	subjects = current_item.subjects
	counts = [subject_count.get(s, 0) for s in subjects]
	raw_score, scaled_score = 0.0, 0.0

	if any(c == 0 for c in counts):
		raw_score = -1.0
		scaled_score = 0.0
	elif all(c >= 2 for c in counts):
		raw_score = 1.0
		scaled_score = 1.0
	elif all(c == 1 for c in counts):
		raw_score = 0.5
		scaled_score = 0.5
	return raw_score, scaled_score

	# Debugging prints
	# print("\nCurrent Item Subjects:", current_item.subjects)
	# print("History Length:", len(history))
	# print("Recent History:", [item.subjects for item in recent_history])
	# print("Subject Count:", subject_count)
	# print("Coherence Score Before Normalization:", coherence_score)
	# print("Coherence Score After Normalization:", coherence_score / len(current_item.subjects) if current_item.subjects else 0.0)
	# print("Number of Subjects in Current Item:", len(current_item.subjects))


def score_freshness(current_item: Item, history: list[Item]) -> float:
	recent_history = history[-6:-2]  # 5 items before current turn
	novel_subjects = 0

	# Check for if we have to account for pauses in the recent history
	history_subjects = set()
	for item in recent_history:
		if item is not None:
			history_subjects.update(item.subjects)

	for subj in current_item.subjects:
		if subj not in history_subjects:
			novel_subjects += 1

	# Should the score be 0.5 or maybe 0.75 for one novel subject?
	if novel_subjects == 0:
		raw_score = 0.0
		scaled_score = 0.0
		return raw_score, scaled_score
	elif novel_subjects == 1:
		raw_score = 1.0
		scaled_score = 0.5
	else:  # novel_subjects = 2
		raw_score = 2.0
		scaled_score = 1.0

	return raw_score, scaled_score


def score_nonmonotonousness(current_item: Item, history: list[Item]) -> float:
	if current_item is None:
		return 0.0

	recent_history = history[-3:]
	penalty = 0

	for subj in current_item.subjects:
		if all(
			prev_item is not None and any(prev_subj == subj for prev_subj in prev_item.subjects)
			for prev_item in recent_history
		):
			penalty -= 1

	if current_item in history:
		penalty -= 1

	raw_score = penalty

	max_penalty = len(current_item.subjects) + 1 if current_item.subjects else 1

	scaled_score = 1.0 - (penalty / max_penalty)  # higher scaled score is more nonmonotonous
	return raw_score, scaled_score


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
	raw_score = sum(bonuses) / len(bonuses)
	scaled_score = raw_score
	return raw_score, scaled_score


def calculate_weighted_score(
	item_id,
	scaled_scores,
	weights,
):
	w1, w2, w3, w4, w5 = weights

	coherence = scaled_scores["coherence"].get(item_id, 0.0)
	importance = scaled_scores["importance"].get(item_id, 0.0)
	preference = scaled_scores["preference"].get(item_id, 0.0)
	nonmonotonousness = scaled_scores["nonmonotonousness"].get(item_id, 0.0)
	freshness = scaled_scores["freshness"].get(item_id, 0.0)

	return (
		w1 * coherence + w2 * importance + w3 * preference + w4 * nonmonotonousness + w5 * freshness
	)


def choose_item(
	weighted: float,
	raw: float,
	memory_bank: list[Item],
	score_sources: dict[str, dict[UUID, tuple[float, float]]],
	weights: tuple[float, float, float, float, float],
):
	
	scaled_scores = {"coherence": {}, "importance": {}, "preference": {}, "nonmonotonousness": {}, "freshness": {}}
	total_raw_scores = {}

	for item in memory_bank:
		item_id = item.id
		raw_score_sum = 0

		for key in score_sources:
			raw_score_sum += score_sources[key][item_id][0]
			scaled_scores[key][item_id] = score_sources[key][item_id][1]

		total_raw_scores[item_id] = raw_score_sum

	total_weighted_scores = {
		item.id: calculate_weighted_score(
			item.id,
			scaled_scores,
			weights
		)
		for item in memory_bank
	}


	final_scores = {item.id: weighted * total_weighted_scores[item.id] + raw * total_raw_scores[item.id] for item in memory_bank}

	if not final_scores:
		return None
	
	# Best candidate now
	best_item_id, best_now = max(final_scores.items(), key=lambda kv: kv[1])
	best_item = next((it for it in memory_bank if it.id == best_item_id), None)

	# Return Best Item and its score, weighted scores for pause decision
	return best_item, best_now, final_scores

	# Takes in the total memory bank and scores each item based on whatever weighting system we have
	# Actually should make this a function in the class so it can have access to the contributed items/memory bank
	# Should automatically score things that were already in the contributed items a 0

	# As its scored, add it to a set thats sorted by the score. Return Set


##################################################
# Helper functions for pause decisions
##################################################


def count_consecutive_pauses(history: list[Item]) -> int:
	# Check only the two most recent moves for consecutive pauses
	cnt = 0
	for it in reversed(history[-2:]):  # Limit to the last two moves
		if it is None:
			cnt += 1
		else:
			break
	return cnt


def _engine_like_turn_impact(history: list[Item], item: Item) -> float:
	"""
	total = importance + coherence + freshness + nonmonotonousness

	We evaluate as-if item were placed at the next index (i = len(history)).
	"""
	if item is None:
		return 0.0

	i = len(history)  # proposed index of this new turn

	# Repetition check across prior history (strict identity by id)
	is_repeated = any(h and h.id == item.id for h in history)

	# --- Importance ---
	if is_repeated:
		importance = 0.0
	else:
		importance = float(getattr(item, "importance", 0.0))

	# Coherence (look back up to 3 until a pause is hit)
	context_items: list[Item] = []
	for j in range(i - 1, max(-1, i - 4), -1):
		if j < 0:
			break
		if history[j] is None:
			break
		context_items.append(history[j])

	context_subject_counts = Counter(
		s for it in context_items for s in (it.subjects if it else [])
	)
	coherence = 0.0
	# If any subject of current item is missing from context, -1
	if not all(s in context_subject_counts for s in item.subjects):
		coherence -= 1.0
	# If all subjects appear at least twice in context, +1
	if item.subjects and all(context_subject_counts.get(s, 0) >= 2 for s in item.subjects):
		coherence += 1.0

	# Freshness (only if previous turn was a pause)
	# Otherwise count novel subjects vs. prior 5 non-None items before the pause.
	if i == 0 or (i > 0 and history[i - 1] is not None):
		freshness = 0.0
	else:
		start = max(0, i - 6)
		prior_items = (h for h in history[start : i - 1] if h is not None)
		prior_subjects = {s for it in prior_items for s in it.subjects}
		novel = [s for s in item.subjects if s not in prior_subjects]
		freshness = float(len(novel))

	# Nonmonotonousness (penalize runs of same-subject in last 3)
	# Engine: if repeated item id then -1.0
	#         elif i < 3 then 0.0
	#         elif all of last 3 items exist and each shares ANY subject with current -> -1.0
	if is_repeated:
		nonmono = -1.0
	elif i < 3:
		nonmono = 0.0
	else:
		last_three = history[max(0, i - 3) : i]
		if last_three and all(
			(h is not None) and any(s in h.subjects for s in item.subjects)
			for h in last_three
		):
			nonmono = -1.0
		else:
			nonmono = 0.0

	# Total (matches engine: excludes any 'individual' bonus)
	total = importance + coherence + freshness + nonmono
	return float(total)


def should_pause(
	history: list[Item],
	best_item: Item,  # <-- add this parameter
) -> bool:
	"""
	Pause iff playing our best item would NOT have a positive turn impact.

	We simulate the engine scoring with engine_like_turn_impact.
	If total > 0 then speak (return False). Otherwise then pause (return True).

	Safety: if we've already paused twice consecutively, lower threshold.
	"""
	# Simulate engine total for the candidate we'd play
	total = _engine_like_turn_impact(history, best_item)

	# Positive impact → speak
	if total > 0.0:
		return False

	# Avoid immediate termination from 3 consecutive pauses:
	cons_pauses = count_consecutive_pauses(history)
	if cons_pauses >= 2:
		# if we're only barely non-positive, speak
		return total <= -0.05

	# Non-positive → pause
	return True
