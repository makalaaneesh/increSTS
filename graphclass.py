class ContextNode(object):
	"""docstring for ContextNode"""
	context_list=[]
	def __init__(self, context_list):
		super(ContextNode, self).__init__()
		self.context_list = context_list
	def __eq__(self, other):
		"""Override the default Equals behavior"""
		if isinstance(other, self.__class__):
		    return set(self.context_list) == set(other.context_list)
		return NotImplemented

	def __ne__(self, other):
		"""Define a non-equality test"""
		if isinstance(other, self.__class__):
		    return not self.__eq__(other)
		return NotImplemented

	def __hash__(self):
		"""Override the default hash behavior (that returns the id or the object)"""
		return hash(str(self.context_list))
	def __repr__(self):
		return str(self.context_list)
	def __str__(self):
		return str(self.context_list)

	def displayNode():
		print context_list

class WordNode(object):
	"""docstring for ContextNode"""
	word=""
	isNoisy=True
	def __init__(self, word,isNoisy):
		super(WordNode, self).__init__()
		self.word = word
		self.isNoisy = isNoisy
	def __eq__(self, other):
		"""Override the default Equals behavior"""
		if isinstance(other, self.__class__):
		    return str(self.word) == str(other.word)
		return NotImplemented

	def __ne__(self, other):
		"""Define a non-equality test"""
		if isinstance(other, self.__class__):
		    return not self.__eq__(other)
		return NotImplemented

	def __hash__(self):
		"""Override the default hash behavior (that returns the id or the object)"""
		return hash(str(self.word))
	def __repr__(self):
		return self.word
	def __str__(self):
		return self.word

	def displayNode():
		print word