#imported necessary packages
from mrjob.job import MRJob 


#Task 1.5 MapReduce Implementation
##determined how many movies were released by each production company for each year

class company_count(MRJob):
    def mapper(self, _, line):
        yield line, 1
        
    def reducer(self, line, count):
        yield line, sum(count)


if __name__ == "__main__":
    company_count.run()