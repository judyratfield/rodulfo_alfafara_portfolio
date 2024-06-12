#imported necessary packages
from mrjob.job import MRJob

#defined class for MRJob
class merge_sort(MRJob):

#defined mapper to distinguish year, company pairs and count portions from the input line
    def mapper(self, _, line):
        year_company, count = line.strip().split('\t')
        count = int(count)
        yield (None, (year_company, count))#set as a tuple because we want to yield these values together as a single unit
        
#defined reducer to perform the mergesort
    def reducer(self, _, year_company_counts):
        sorted_data = sorted(year_company_counts,#applied the sorted function to automatically employ Mergesort for sorting
                             key=lambda p: p[1],#p is the tuple(year_company, count) while p[1] selects the second element of the tuple, count to be the point of comparison when sorting.
                             reverse= False)#reverse is set to False as sorting should be in ascending order
        for year_company, count in sorted_data:
            yield (year_company.strip('"'), count)


if __name__== "__main__":
    merge_sort.run()