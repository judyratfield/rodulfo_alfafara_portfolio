#imported necessary packages
import pymongo
 

#assigned connection to client variable
client = pymongo.MongoClient("mongodb://localhost:27017/")

#checked database names
db_names = client.list_database_names()
# print(db_names)

#assigned database to be used to variable db
##database has been created via Studio3T
db = client['Assignment1'] 

#checked collection entitled movies can be accessed
##collection movies has been imported to the database above via Studio3T
col_names = db.list_collection_names()
# print(col_names)

#assigned the imported collection on movies to movies variable
movies = db['movies']

#checked that documents contained from the collection can be accessed
# for doc in movies.find():
#     print(doc)

#Task 1.1 Data Retrieval
##Extracted the year from each document
##updated the date to reflect only the year
for doc in movies.find():
    year = doc['date'][-4:]
    doc_id = doc['_id']
    db.movies.update_one({"_id":doc_id},{"$set":{"date":year}})

#Task 1.2 Company Identification
##updated companies to reflect only the first 3 companies

for doc in movies.find():
    doc_id = doc['_id']
    company_list = []
    for company in doc['companies']:
        if len(company_list) < 3:
            company_list.append(company)
    db.movies.update_one({"_id":doc_id},{"$set":{"companies":company_list}})

#Task 1.3 Data Formatting
##Generated a series of pairs for each movie
##Used the format: <year,company>

year_company = []
for doc in movies.find():
    for company in doc['companies']:
        name = company['name']
        pair = f"{doc['date']}, {name}"
        year_company.append(pair)
for yc_pair in year_company:
    print(yc_pair)


#Task 1.4 Data Storage
#Stored <year, company> pairs of all movies into a text file
##Did this by running the command below via command prompt
###C:\Users\rodul\OneDrive\Desktop\Macquarie\second_sem_2023\COMP_6210_Big_Data\Assignments>python task1_extraction.py >year_and_company.txt

