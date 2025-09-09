from models.player import Item, Player, PlayerSnapshot
import uuid


class Player1(Player):
	def __init__(self, snapshot: PlayerSnapshot, conversation_length: int) -> None:  # noqa: F821
		super().__init__(snapshot, conversation_length)
	
	def propose_item(self, history: list[Item]) -> Item | None:

		print("\nCurrent Memory Bank: ", self.memory_bank)
		print("\nConversation History: ", history)

		
		# If history length is 0, return the first item from preferred sort ( Has the highest Importance)
		if len(history) == 0:
			memory_bank_imp = importanceSort(self.memory_bank)
			# print("Sorted by Important: ", memory_bank_imp)
			# input("Press Enter to continue... Importance Sort Complete")
			return memory_bank_imp[0] if memory_bank_imp else None

		# This Checks Repitition so we dont repeat any item that has already been said in the history, returns a filtered memory bank
		filtered_memory_bank = checkRepition(history, self.UsedItems, self.memory_bank)
		print("\nFiltered Memory Bank: ", filtered_memory_bank)

		# Return None if there are no valid items to propose
		# This can be changed in future just incase we run out of things to say and have to repeat. Not sure if this is possible
		if len(filtered_memory_bank) == 0:
			print("No valid items to propose after filtering")
			# Use importance score if no items are left after filtering
			filtered_memory_bank = importanceSort(self.memory_bank)

		# Sort memory bank based on coherence and importanceSort
		memory_bank_co = coherenceSort(filtered_memory_bank, history)
		# print("Sorted by Coherence: ", memory_bank_co)
		# input("Press Enter to continue... Coherence Sort Complete")

		memory_bank_imp = importanceSort(filtered_memory_bank)


		# TODO: Add Preferred Sort back in once implemented
		# memory_bank_pref = preferredSort(filtered_memory_bank)

		# TODO: Finish Weight Matrix
		# weighted_list = weightMatrix(filtered_memory_bank, memory_bank_co, memory_bank_imp, memory_bank_pref)

		# I just have it returning the highest coherence item for now
		if memory_bank_co:
			return memory_bank_co[0]
		return None

	#Personal Variables
	LastSuggestion: Item
	UsedItems: set[uuid.UUID] = set()

# Helper Functions #

def checkRepition(history: list[Item], UsedItems, memory_bank) -> list[Item]:
	# Update the proposed items set with items from history
	UsedItems.update(item.id for item in history)

	# Filter out items with IDs already in the proposed items set
	return [item for item in memory_bank if item.id not in UsedItems]

def coherenceCheck(currentItem: Item, history: list[Item]) -> float:
	# Check the last 3 items in history (or fewer if history is shorter)
	recent_history = history[-3:]
	coherence_score = 0

	# Count occurrences of each subject in the recent history
	subject_count = {}
	for item in recent_history:
		for subject in item.subjects:
			subject_count[subject] = subject_count.get(subject, 0) + 1

	# See if all subjects in the current item are appear once or twice in the history
	for subject in currentItem.subjects:
		if subject_count.get(subject, 0) in [1, 2]:
			coherence_score += 1

	# Debugging prints
	# print("\nCurrent Item Subjects:", currentItem.subjects)
	# print("History Length:", len(history))
	# print("Recent History:", [item.subjects for item in recent_history])
	# print("Subject Count:", subject_count)
	# print("Coherence Score Before Normalization:", coherence_score)
	# print("Coherence Score After Normalization:", coherence_score / len(currentItem.subjects) if currentItem.subjects else 0.0)
	# print("Number of Subjects in Current Item:", len(currentItem.subjects))


	# This should return a score between 0 and 1 (Not exactly the 0 .5 1 you wanted can be changed later)
	return coherence_score / len(currentItem.subjects) if currentItem.subjects else 0.0

def coherenceSort(memory_bank: list[Item], history: list[Item]) -> list[Item]:
	# Sort the memory bank based on coherence scores in descending order
	# use a lambda on each item to check coherence score
	sorted_memory = sorted(
		memory_bank,
		key=lambda item: coherenceCheck(item, history),
		reverse=True
	)
	return sorted_memory

def importanceSort(memory_bank: list[Item]) -> list[Item]:
	# Sort the memory bank based on the importance attribute in descending order
	return sorted(memory_bank, key=lambda item: item.importance, reverse=True)

def preferredSort (memory_bank: list[Item]):
    #Returns a list of the memory bank based on preference sorting 
	return None

def weightMatrix (memory_bank: list[Item], coherence: list[Item], importance: list[Item], preference: list[Item]):
	#Takes in the total memory bank and scores each item based on whatever weighting system we have
	#Actually should make this a function in the class so it can have access to the contributed items/memory bank
	#Should automatically score things that were already in the contributed items a 0

	#As its scored, add it to a set thats sorted by the score. Return Set
	return None