import pp
import math
import time


ppservers = ("*", )


job_server =  pp.Server(ppservers = ppservers)



def print_available(job_server):
	for computer, cpu_count in job_server.get_active_nodes().iteritems():
		print "Found {} with CPU count {}!".format(computer, cpu_count)


def isprime(n):
	"""Returns True if n is prime and False otherwise"""
	# print "Isprime"
	if n < 2:
		return False
	if n == 2:
		return True
	# no reason to go through all the numbers; the square root is as far as
	# we'll get anyway
	max = int(math.ceil(math.sqrt(n)))
	i = 2
	while i <= max:
		if n % i == 0:
			return False
		i += 1
	return True


def sum_primes(n):
	"""Calculates sum of all primes below given integer n"""
	return sum([x for x in xrange(2, n) if isprime(x)])


time.sleep(5)
print_available(job_server)


inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700)
# inputs = (100,200,300,400,500,600,700,800)
jobs = [(input, job_server.submit(sum_primes, (input,), (isprime,), ("math",))) for input in inputs]
print jobs
for input, job in jobs:
	print "Sum of primes below", input, "is", job()

# print sum_primes(5)

# print sum_primes(20)
