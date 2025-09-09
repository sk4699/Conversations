from models.player import Item, Player, PlayerSnapshot

class Player1(Player):
	def __init__(self, snapshot: PlayerSnapshot, conversation_length: int) -> None:  # noqa: F821
		super().__init__(snapshot, conversation_length)
	
	def propose_item(self, history: list[Item]) -> Item | None:
		#To start check the current item in history and see if it matches LAST SUGGESTION (add to contributed item list)

		if (history[-1].id == self.LastSuggestion.id):
			self.contributed_items.append(history[-1])
		
		#Now when considering an option we can always check it with the contributed items list

		#Creating a list sorted based on importance and preference
		memory_bank_imp = sorted(self.memory_bank, key=lambda x: x.count, reverse=True)
		memory_bank_pref = preferredSort(self.memory_bank)
		memory_bank_co = coherenceSort(self.memory_bank)

		weighted_list = weightMatrix(self.memory_bank, memory_bank_co, memory_bank_imp, memory_bank_pref)
		
		#From here choose the highest one and return the item (Assign this as LastSuggestion)

		return None
	

	#Personal Variables
	LastSuggestion: Item

# Helper Functions #

def coherenceCheck(currentItem: Item, history: list[Item]):
	#Function that checks the current history (past 3 items) to see if current item is coherent
	#Returns 0 (not coherent), 0.5 (coherent with half of the items subjects), 1 (coherent totally)  
	return None

def coherenceSort (memory_bank: list[Item]):
	#Performs Coherence Check on the memory bank and sorts it into a descending list 1>0.5>0
	#Returns a list
	return None

def preferredSort (memory_bank: list[Item]):
    #Returns a list of the memory bank based on preference sorting 
	return None

def weightMatrix (memory_bank: list[Item], coherence: list[Item], importance: list[Item], preference: list[Item]):
	#Takes in the total memory bank and scores each item based on whatever weighting system we have
	#Actually should make this a function in the class so it can have access to the contributed items/memory bank
	#Should automatically score things that were already in the contributed items a 0

	#As its scored, add it to a set thats sorted by the score. Return Set
	return None