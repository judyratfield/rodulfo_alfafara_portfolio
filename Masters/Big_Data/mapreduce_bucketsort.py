#imported necessary packages
from mrjob.job import MRJob
from mrjob.step import MRStep

#For this task, we will be using a bucket size of 3 which means that our range will be as follows:
#range (3x, 3x+3]; where x is the bucket's index. To clarify, bucket 1 has index 0, bucket 2 has index 1, and so on...
##bucket 1 will have a range of (0,3], bucket 2 (3,6], bucket 3 (6,9], bucket 4 (9,12], bucket 5 (12,15]
##We stop at bucket 5 because our max count value is only 15 based on Task 2.1 (mergesort)

#defined class for MRJob
class bucket_sort(MRJob):
    #defined steps since we will have more than one mapper and reducer
    def steps(self): 
        return [MRStep(mapper=self.mapper1, 
                        reducer=self.reducer1), 
                MRStep(reducer=self.reducer2)]
    

#defined a mapper that assigns bucket ids to year_company and count pairs
#Assigned each pair based on the range specified above

    def mapper1(self, _, line):
        bucket_id = 0
        year_company, count = line.strip().split('\t')
        count = int(count)
        if count <= 3:
            bucket_id = 1
            yield (None, (year_company, count, bucket_id))
        elif count > 3 and count <= 6:
            bucket_id = 2
            yield (None, (year_company, count, bucket_id))
        elif count > 6 and count <= 9:
            bucket_id = 3
            yield (None, (year_company, count, bucket_id))
        elif count > 9 and count <= 12:
            bucket_id = 4
            yield (None, (year_company, count, bucket_id))
        elif count > 12 and count <= 15:
            bucket_id = 5
            yield (None, (year_company, count, bucket_id))


#defined a reducer that sorts the bucket ids in descending order          
    def reducer1(self, _, year_company_counts_bucket):
        sorted_data = sorted(year_company_counts_bucket,
                              key=lambda p: p[2],#p is the tuple(year_company, count, bucket_id) while p[2] selects the third element of the tuple, bucket_id to be the point of comparison when sorting.
                              reverse= True)#reverse is set to True as sorting should be in descending order
        for year_company, count, bucket_id in sorted_data:
            yield None,(year_company.strip('"'), count, bucket_id)


#defined a reducer that sorts the pairs within each bucket according to their counts in descending order
    def reducer2(self, _, year_company_counts_bucket):
        sorted_data = sorted(year_company_counts_bucket,
                              key=lambda p: p[1],#p is the tuple(year_company, count, bucket_id) while p[1] selects the second element of the tuple, count to be the point of comparison when sorting.
                              reverse= True)#reverse is set to True as sorting should be in descending order
        for year_company, count, bucket_id in sorted_data:
            yield (year_company.strip('"'), count)


if __name__== "__main__":
    bucket_sort.run()